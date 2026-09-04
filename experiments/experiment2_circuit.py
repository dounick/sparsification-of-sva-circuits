import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.circuit import capture_scope, mean_scope_activations, run_intervention
from src.data import gap_values, load_data, prepare_batch


def evaluate(model, tokenizer, data, means, device, batch_size):
    values = {
        "clean": [],
        "source": [],
        "circuit_mean": [],
        "circuit_source": [],
        "scope_mean": [],
        "scope_source": [],
    }

    for start in range(0, len(data), batch_size):
        rows = data[start : start + batch_size]
        base, source, subject, correct, incorrect = prepare_batch(
            rows, tokenizer, device
        )
        base_prediction = (base["attention_mask"].sum(1) - 1).tolist()
        source_prediction = (source["attention_mask"].sum(1) - 1).tolist()
        source_activations, source_logits = capture_scope(
            model, source, subject, source_prediction
        )
        with torch.inference_mode():
            clean_logits = model(**base).logits

        outputs = {
            "clean": clean_logits,
            "source": source_logits,
            "circuit_mean": run_intervention(
                model, base, subject, base_prediction, means, mode="circuit"
            ),
            "circuit_source": run_intervention(
                model,
                base,
                subject,
                base_prediction,
                source_activations,
                mode="circuit",
            ),
            "scope_mean": run_intervention(
                model, base, subject, base_prediction, means, mode="scope"
            ),
            "scope_source": run_intervention(
                model,
                base,
                subject,
                base_prediction,
                source_activations,
                mode="scope",
            ),
        }
        for name, logits in outputs.items():
            mask = source["attention_mask"] if name == "source" else base["attention_mask"]
            values[name].extend(
                gap_values(logits, mask, correct, incorrect).cpu().tolist()
            )
        print(f"Circuit {start + len(rows)}/{len(data)}", flush=True)

    return {key: np.asarray(value, dtype=np.float32) for key, value in values.items()}


def summarize(values):
    means = {key: float(value.mean()) for key, value in values.items()}
    clean = means["clean"]
    source = means["source"]
    return {
        **{f"{key}_gap": value for key, value in means.items()},
        "circuit_necessity": (clean - means["circuit_mean"]) / clean,
        "circuit_sufficiency": (clean - means["circuit_source"]) / (clean - source),
        "scope_necessity": (clean - means["scope_mean"]) / clean,
        "scope_sufficiency": (clean - means["scope_source"]) / (clean - source),
    }


def cache_focals(model, tokenizer, data, device, batch_size):
    cached = []
    for start in range(0, len(data), batch_size):
        rows = data[start : start + batch_size]
        base, _, subject, correct, incorrect = prepare_batch(rows, tokenizer, device)
        prediction = (base["attention_mask"].sum(1) - 1).tolist()
        activations, logits = capture_scope(model, base, subject, prediction)
        gaps = gap_values(logits, base["attention_mask"], correct, incorrect)
        for i, row in enumerate(rows):
            cached.append(
                {
                    "inputs": {key: value[i : i + 1].cpu() for key, value in base.items()},
                    "subject": subject[i],
                    "prediction": prediction[i],
                    "correct": int(correct[i].item()),
                    "incorrect": int(incorrect[i].item()),
                    "number": row["base_number"],
                    "gap": float(gaps[i]),
                    "activations": {
                        key: value[i].cpu() for key, value in activations.items()
                    },
                }
            )
    return cached


def off_circuit(model, tokenizer, data, device, batch_size, n_donors, seed):
    cached = cache_focals(model, tokenizer, data, device, batch_size)
    rng = np.random.RandomState(seed)
    gaps = []
    focal_indices = []
    donor_indices = []
    same_number = []
    focal_numbers = []

    for focal_index, focal in enumerate(cached):
        candidates = [i for i in range(len(cached)) if i != focal_index]
        donors = rng.choice(candidates, n_donors, replace=False)
        inputs = {
            key: value.to(device).repeat((len(donors),) + (1,) * (value.ndim - 1))
            for key, value in focal["inputs"].items()
        }
        donor_activations = {
            key: torch.stack([cached[i]["activations"][key] for i in donors]).to(device)
            for key in focal["activations"]
        }
        focal_activations = {
            key: value.to(device)[None, :].expand(len(donors), -1)
            for key, value in focal["activations"].items()
        }
        logits = run_intervention(
            model,
            inputs,
            [focal["subject"]] * len(donors),
            [focal["prediction"]] * len(donors),
            donor_activations,
            mode="off_circuit",
            focal=focal_activations,
        )
        correct = torch.full(
            (len(donors),), focal["correct"], device=device, dtype=torch.long
        )
        incorrect = torch.full(
            (len(donors),), focal["incorrect"], device=device, dtype=torch.long
        )
        batch_gaps = gap_values(
            logits, inputs["attention_mask"], correct, incorrect
        ).cpu().tolist()
        for donor, gap in zip(donors, batch_gaps):
            focal_indices.append(focal_index)
            donor_indices.append(int(donor))
            same_number.append(cached[donor]["number"] == focal["number"])
            focal_numbers.append(focal["number"])
            gaps.append(gap)
        print(f"Off-circuit {focal_index + 1}/{len(cached)}", flush=True)

    gaps = np.asarray(gaps, dtype=np.float32)
    same_number = np.asarray(same_number, dtype=bool)
    focal_numbers = np.asarray(focal_numbers)
    return {
        "baseline_gap": np.asarray([row["gap"] for row in cached], dtype=np.float32),
        "baseline_number": np.asarray([row["number"] for row in cached]),
        "gap": gaps,
        "focal_index": np.asarray(focal_indices, dtype=np.int32),
        "donor_index": np.asarray(donor_indices, dtype=np.int32),
        "same_number": same_number,
        "focal_number": focal_numbers,
        "same_number_gap": float(gaps[same_number].mean()),
        "cross_number_gap": float(gaps[~same_number].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument("--checkpoint", type=int, default=143000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=f"step{args.checkpoint}",
        attn_implementation="eager",
    ).to(args.device).eval()

    train_data = load_data(ROOT / "data/pp_train.jsonl")
    eval_data = load_data(ROOT / "data/pp_eval.jsonl")
    batch_size = 4
    means = mean_scope_activations(
        model, tokenizer, train_data, args.device, batch_size
    )
    values = evaluate(
        model, tokenizer, eval_data, means, args.device, batch_size
    )
    summary = summarize(values)
    summary.update(
        {"model": args.model, "checkpoint": args.checkpoint, "n": len(eval_data)}
    )
    off = off_circuit(
        model, tokenizer, eval_data, args.device, batch_size, n_donors=20, seed=123
    )

    model_name = args.model.split("/")[-1]
    output_dir = ROOT / "results/experiment2_circuit" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"step{args.checkpoint}.npz"
    np.savez(
        raw_path,
        **values,
        off_circuit_baseline_gap=off["baseline_gap"],
        off_circuit_gap=off["gap"],
        off_circuit_focal_index=off["focal_index"],
        off_circuit_donor_index=off["donor_index"],
        off_circuit_same_number=off["same_number"],
        off_circuit_focal_number=off["focal_number"],
    )
    baseline_singular = off["baseline_number"] == "singular"
    trial_singular = off["focal_number"] == "singular"
    summary["off_circuit"] = {
        "n_focal": len(eval_data),
        "n_donors": 20,
        "baseline_gap": float(off["baseline_gap"].mean()),
        "baseline_singular_gap": float(off["baseline_gap"][baseline_singular].mean()),
        "baseline_plural_gap": float(off["baseline_gap"][~baseline_singular].mean()),
        "same_number_n": int(off["same_number"].sum()),
        "cross_number_n": int((~off["same_number"]).sum()),
        "same_number_gap": off["same_number_gap"],
        "cross_number_gap": off["cross_number_gap"],
        "same_number_singular_gap": float(
            off["gap"][off["same_number"] & trial_singular].mean()
        ),
        "same_number_plural_gap": float(
            off["gap"][off["same_number"] & ~trial_singular].mean()
        ),
        "cross_number_singular_gap": float(
            off["gap"][~off["same_number"] & trial_singular].mean()
        ),
        "cross_number_plural_gap": float(
            off["gap"][~off["same_number"] & ~trial_singular].mean()
        ),
    }
    summary_path = output_dir / f"step{args.checkpoint}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"clean={summary['clean_gap']:+.4f} source={summary['source_gap']:+.4f}")
    print(
        f"circuit mean={summary['circuit_mean_gap']:+.4f} "
        f"necessity={summary['circuit_necessity']:.1%} "
        f"source={summary['circuit_source_gap']:+.4f} "
        f"sufficiency={summary['circuit_sufficiency']:.1%}"
    )
    print(
        f"scope mean={summary['scope_mean_gap']:+.4f} "
        f"necessity={summary['scope_necessity']:.1%} "
        f"source={summary['scope_source_gap']:+.4f} "
        f"sufficiency={summary['scope_sufficiency']:.1%}"
    )
    print(
        f"off-circuit same={off['same_number_gap']:+.4f} "
        f"cross={off['cross_number_gap']:+.4f}"
    )
    print(summary_path)


if __name__ == "__main__":
    main()
