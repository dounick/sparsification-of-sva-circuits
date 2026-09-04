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
from src.data import gap_values, load_data, prepare_batch, prepare_inputs


TEMPLATES = {
    "pp": "pp_eval.jsonl",
    "subject_rc": "subject_rc_eval.jsonl",
    "object_rc": "object_rc_eval.jsonl",
    "simple": "simple_eval.jsonl",
    "wikitext": "wikitext_eval.jsonl",
}

PERTURBATIONS = {
    "adjective_subject": "pp_adjective_subject_eval.jsonl",
    "adjective_object": "pp_adjective_object_eval.jsonl",
    "adverb": "pp_adverb_eval.jsonl",
    "single_prefix": "pp_single_prefix_eval.jsonl",
    "locative_prefix": "pp_locative_prefix_eval.jsonl",
    "long_distractor": "pp_long_distractor_eval.jsonl",
}


def metrics(clean, source, intervention, ablation=None):
    clean_mean = float(np.mean(clean))
    source_mean = float(np.mean(source))
    intervention_mean = float(np.mean(intervention))
    result = {
        "n": len(clean),
        "clean_gap": clean_mean,
        "source_gap": source_mean,
        "intervention_gap": intervention_mean,
        "sufficiency": (clean_mean - intervention_mean) / (clean_mean - source_mean),
        "flipped_n": int(np.sum(np.asarray(intervention) < 0)),
        "flipped_fraction": float(np.mean(np.asarray(intervention) < 0)),
        "per_example": {
            "clean_gap": np.asarray(clean).tolist(),
            "source_gap": np.asarray(source).tolist(),
            "intervention_gap": np.asarray(intervention).tolist(),
        },
    }
    if ablation is not None:
        ablation_mean = float(np.mean(ablation))
        result["ablation_gap"] = ablation_mean
        result["necessity"] = (clean_mean - ablation_mean) / clean_mean
        result["per_example"]["ablation_gap"] = np.asarray(ablation).tolist()
    return result


def evaluate_rows(model, tokenizer, data, device, batch_size, means=None):
    clean_gaps = []
    source_gaps = []
    intervention_gaps = []
    ablation_gaps = []

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
        intervention_logits = run_intervention(
            model,
            base,
            subject,
            base_prediction,
            source_activations,
            mode="circuit",
        )
        clean_gaps.extend(
            gap_values(clean_logits, base["attention_mask"], correct, incorrect)
            .cpu()
            .tolist()
        )
        source_gaps.extend(
            gap_values(source_logits, source["attention_mask"], correct, incorrect)
            .cpu()
            .tolist()
        )
        intervention_gaps.extend(
            gap_values(
                intervention_logits, base["attention_mask"], correct, incorrect
            )
            .cpu()
            .tolist()
        )
        if means is not None:
            ablation_logits = run_intervention(
                model, base, subject, base_prediction, means, mode="circuit"
            )
            ablation_gaps.extend(
                gap_values(
                    ablation_logits, base["attention_mask"], correct, incorrect
                )
                .cpu()
                .tolist()
            )

    return metrics(
        clean_gaps,
        source_gaps,
        intervention_gaps,
        ablation_gaps if means is not None else None,
    )


def choose_donors(base_data, donor_data, seed=43):
    rng = np.random.RandomState(seed)
    pools = {
        number: [i for i, row in enumerate(donor_data) if row["base_number"] == number]
        for number in ("singular", "plural")
    }
    for pool in pools.values():
        rng.shuffle(pool)
    next_index = {"singular": 0, "plural": 0}
    selected = []
    for base_row in base_data:
        number = "plural" if base_row["base_number"] == "singular" else "singular"
        pool = pools[number]
        donor_index = pool[next_index[number] % len(pool)]
        next_index[number] += 1
        donor = donor_data[donor_index]
        selected.append(
            {
                "base": donor["base"],
                "source": donor["source"],
                "donor_index": donor_index,
            }
        )
    return selected


def evaluate_cross_template(
    model, tokenizer, base_data, donor_data, device, batch_size
):
    donors = choose_donors(base_data, donor_data)
    clean_gaps = []
    source_gaps = []
    intervention_gaps = []

    for start in range(0, len(base_data), batch_size):
        base_rows = base_data[start : start + batch_size]
        donor_rows = donors[start : start + batch_size]
        base, source, subject, correct, incorrect = prepare_batch(
            base_rows, tokenizer, device
        )
        donor, _, donor_subject = prepare_inputs(donor_rows, tokenizer, device)
        base_prediction = (base["attention_mask"].sum(1) - 1).tolist()
        source_prediction = (source["attention_mask"].sum(1) - 1).tolist()
        donor_prediction = (donor["attention_mask"].sum(1) - 1).tolist()
        donor_activations, _ = capture_scope(
            model, donor, donor_subject, donor_prediction
        )
        with torch.inference_mode():
            clean_logits = model(**base).logits
            source_logits = model(**source).logits
        intervention_logits = run_intervention(
            model,
            base,
            subject,
            base_prediction,
            donor_activations,
            mode="circuit",
        )
        clean_gaps.extend(
            gap_values(clean_logits, base["attention_mask"], correct, incorrect)
            .cpu()
            .tolist()
        )
        source_gaps.extend(
            gap_values(source_logits, source["attention_mask"], correct, incorrect)
            .cpu()
            .tolist()
        )
        intervention_gaps.extend(
            gap_values(
                intervention_logits, base["attention_mask"], correct, incorrect
            )
            .cpu()
            .tolist()
        )

    result = metrics(clean_gaps, source_gaps, intervention_gaps)
    result["donors"] = [row["donor_index"] for row in donors]
    return result


def encode_pairs(tokenizer, pairs):
    encoded = []
    for pair in pairs:
        singular = tokenizer(pair["singular"], add_special_tokens=False)["input_ids"]
        plural = tokenizer(pair["plural"], add_special_tokens=False)["input_ids"]
        if len(singular) != 1 or len(plural) != 1:
            raise ValueError(pair)
        encoded.append((singular[0], plural[0]))
    return encoded


def lexical_gaps(logits, attention_mask, numbers, token_pairs):
    rows = torch.arange(logits.shape[0], device=logits.device)
    positions = attention_mask.sum(1) - 1
    last = logits[rows, positions]
    gaps = torch.empty(len(numbers), len(token_pairs), device=logits.device)
    for i, number in enumerate(numbers):
        for j, (singular, plural) in enumerate(token_pairs):
            correct, incorrect = (
                (singular, plural) if number == "singular" else (plural, singular)
            )
            gaps[i, j] = last[i, correct] - last[i, incorrect]
    return gaps


def evaluate_lexical(model, tokenizer, data, token_pairs, device, batch_size):
    clean = []
    source = []
    intervention = []
    for start in range(0, len(data), batch_size):
        rows = data[start : start + batch_size]
        base, source_inputs, subject = prepare_inputs(rows, tokenizer, device)
        base_prediction = (base["attention_mask"].sum(1) - 1).tolist()
        source_prediction = (source_inputs["attention_mask"].sum(1) - 1).tolist()
        source_activations, source_logits = capture_scope(
            model, source_inputs, subject, source_prediction
        )
        with torch.inference_mode():
            clean_logits = model(**base).logits
        intervention_logits = run_intervention(
            model,
            base,
            subject,
            base_prediction,
            source_activations,
            mode="circuit",
        )
        numbers = [row["base_number"] for row in rows]
        clean.append(lexical_gaps(clean_logits, base["attention_mask"], numbers, token_pairs).cpu())
        source.append(lexical_gaps(source_logits, source_inputs["attention_mask"], numbers, token_pairs).cpu())
        intervention.append(lexical_gaps(intervention_logits, base["attention_mask"], numbers, token_pairs).cpu())

    clean = torch.cat(clean).numpy()
    source = torch.cat(source).numpy()
    intervention = torch.cat(intervention).numpy()
    result = metrics(clean.reshape(-1), source.reshape(-1), intervention.reshape(-1))
    result["n"] = len(data)
    result["n_verb_pairs"] = len(token_pairs)
    result["per_example"] = {
        "clean_gap": clean.tolist(),
        "source_gap": source.tolist(),
        "intervention_gap": intervention.tolist(),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument("--checkpoint", type=int, default=143000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["templates", "cross-template", "perturbations", "lexical"],
        default=["templates", "cross-template", "perturbations", "lexical"],
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=f"step{args.checkpoint}",
        attn_implementation="eager",
    ).to(args.device).eval()
    batch_size = 4
    data = {name: load_data(ROOT / "data" / path) for name, path in TEMPLATES.items()}
    results = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "tests": args.tests,
    }

    if "templates" in args.tests:
        print("\nTemplates", flush=True)
        pp_means = mean_scope_activations(
            model,
            tokenizer,
            load_data(ROOT / "data/pp_train.jsonl"),
            args.device,
            batch_size,
        )
        results["templates"] = {}
        for name, rows in data.items():
            means = pp_means if name == "pp" else mean_scope_activations(
                model, tokenizer, rows, args.device, batch_size
            )
            result = evaluate_rows(
                model, tokenizer, rows, args.device, batch_size, means
            )
            results["templates"][name] = result
            print(
                f"{name}: clean={result['clean_gap']:+.4f} "
                f"mean={result['ablation_gap']:+.4f} "
                f"necessity={result['necessity']:.1%} "
                f"source={result['intervention_gap']:+.4f} "
                f"sufficiency={result['sufficiency']:.1%}",
                flush=True,
            )

    if "cross-template" in args.tests:
        print("\nCross-template", flush=True)
        names = ["pp", "subject_rc", "object_rc", "simple"]
        results["cross_template"] = {}
        for base_name in names:
            results["cross_template"][base_name] = {}
            for donor_name in names:
                result = evaluate_cross_template(
                    model,
                    tokenizer,
                    data[base_name],
                    data[donor_name],
                    args.device,
                    batch_size,
                )
                results["cross_template"][base_name][donor_name] = result
                print(
                    f"{base_name} <- {donor_name}: "
                    f"gap={result['intervention_gap']:+.4f} "
                    f"sufficiency={result['sufficiency']:.1%}",
                    flush=True,
                )

    if "perturbations" in args.tests:
        print("\nPerturbations", flush=True)
        results["perturbations"] = {}
        for name, filename in PERTURBATIONS.items():
            rows = load_data(ROOT / "data" / filename)
            result = evaluate_rows(
                model, tokenizer, rows, args.device, batch_size
            )
            results["perturbations"][name] = result
            print(
                f"{name}: gap={result['intervention_gap']:+.4f} "
                f"sufficiency={result['sufficiency']:.1%}",
                flush=True,
            )

    if "lexical" in args.tests:
        print("\nLexical", flush=True)
        copulas = encode_pairs(tokenizer, load_data(ROOT / "data/copulas.jsonl"))
        heldout_verbs = encode_pairs(
            tokenizer, load_data(ROOT / "data/heldout_verbs.jsonl")
        )
        lexical = {
            "heldout_subjects": (
                load_data(ROOT / "data/heldout_subject_eval.jsonl"), copulas
            ),
            "heldout_verbs": (
                load_data(ROOT / "data/heldout_verb_eval.jsonl"), heldout_verbs
            ),
        }
        results["lexical"] = {}
        for name, (rows, token_pairs) in lexical.items():
            result = evaluate_lexical(
                model, tokenizer, rows, token_pairs, args.device, batch_size
            )
            results["lexical"][name] = result
            print(
                f"{name}: gap={result['intervention_gap']:+.4f} "
                f"sufficiency={result['sufficiency']:.1%}",
                flush=True,
            )

    model_name = args.model.split("/")[-1]
    output = (
        ROOT
        / "results/experiment2_generalization"
        / model_name
        / f"step{args.checkpoint}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(output)


if __name__ == "__main__":
    main()
