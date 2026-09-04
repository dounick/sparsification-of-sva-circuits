import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.circuit import TRANSPORT_HEAD
from src.data import gap_values, load_data, prepare_batch


def locations(head, source_positions, base_positions):
    source = ([[head]] * len(source_positions), [[p] for p in source_positions])
    base = ([[head]] * len(base_positions), [[p] for p in base_positions])
    return [[source], [base]]


def train_direction(model, tokenizer, data, device, seed):
    from pyvene import (
        IntervenableConfig,
        IntervenableModel,
        LowRankRotatedSpaceIntervention,
        RepresentationConfig,
        set_seed,
    )

    layer, head = TRANSPORT_HEAD
    set_seed(seed)
    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(
                layer=layer,
                component="head_attention_value_output",
                unit="h.pos",
                low_rank_dimension=1,
            )
        ],
        intervention_types=LowRankRotatedSpaceIntervention,
    )
    intervenable = IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    optimizer = torch.optim.Adam(intervenable.get_trainable_parameters(), lr=5e-3)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        for start in range(0, len(data), 4):
            rows = data[start : start + 4]
            base, source, _, _, source_labels = prepare_batch(
                rows, tokenizer, device
            )
            base_positions = (base["attention_mask"].sum(1) - 1).tolist()
            source_positions = (source["attention_mask"].sum(1) - 1).tolist()
            optimizer.zero_grad()
            _, output = intervenable(
                base,
                [source],
                {
                    "sources->base": locations(
                        head, source_positions, base_positions
                    )
                },
                output_original_output=True,
            )
            batch_rows = torch.arange(len(rows), device=device)
            logits = output.logits[
                batch_rows, torch.as_tensor(base_positions, device=device)
            ]
            loss = loss_function(logits, source_labels)
            loss.backward()
            optimizer.step()
        print(f"DAS seed={seed} epoch={epoch + 1}/5", flush=True)

    vector = None
    for intervention in intervenable.interventions.values():
        vector = intervention.rotate_layer.weight.detach().cpu().squeeze()
        vector = vector / vector.norm()
        break
    del intervenable
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vector


def capture_head(model, inputs, prediction_positions):
    layer, head = TRANSPORT_HEAD
    stored = {}

    def hook(module, args):
        stored["value"] = args[0].detach()
        return args

    handle = model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            logits = model(**inputs).logits
    finally:
        handle.remove()

    device = inputs["input_ids"].device
    rows = torch.arange(inputs["input_ids"].shape[0], device=device)
    positions = torch.as_tensor(prediction_positions, device=device)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    left = head * head_dim
    right = left + head_dim
    return stored["value"][rows, positions, left:right].clone(), logits


def mean_head(model, tokenizer, data, device):
    total = None
    count = 0
    for start in range(0, len(data), 4):
        rows = data[start : start + 4]
        base, _, _, _, _ = prepare_batch(rows, tokenizer, device)
        positions = (base["attention_mask"].sum(1) - 1).tolist()
        values, _ = capture_head(model, base, positions)
        if total is None:
            total = torch.zeros(values.shape[1], dtype=torch.float64)
        total += values.sum(0).cpu().to(torch.float64)
        count += len(rows)
    return (total / count).to(device=device, dtype=torch.float32)


def run_head_intervention(model, inputs, positions, target, mode, vector=None):
    layer, head = TRANSPORT_HEAD
    device = inputs["input_ids"].device
    rows = torch.arange(inputs["input_ids"].shape[0], device=device)
    positions = torch.as_tensor(positions, device=device)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    left = head * head_dim
    right = left + head_dim
    if target.ndim == 1:
        target = target[None, :].expand(len(rows), -1)
    target = target.to(device)
    if vector is not None:
        vector = vector.to(device)

    def hook(module, args):
        activation = args[0].clone()
        natural = activation[rows, positions, left:right]
        if mode == "full":
            replacement = target
        elif mode == "one_d":
            difference = ((target - natural) @ vector).unsqueeze(1)
            replacement = natural + difference * vector
        elif mode == "complement":
            difference = ((natural - target) @ vector).unsqueeze(1)
            replacement = target + difference * vector
        else:
            raise ValueError(mode)
        activation[rows, positions, left:right] = replacement.to(activation.dtype)
        return (activation,)

    handle = model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            return model(**inputs).logits
    finally:
        handle.remove()


def evaluate(model, tokenizer, data, mean, vectors, device):
    raw = {
        "clean": [],
        "source": [],
        "full_mean": [],
        "full_source": [],
        "one_d_mean": [[] for _ in vectors],
        "one_d_source": [[] for _ in vectors],
        "complement_mean": [[] for _ in vectors],
        "complement_source": [[] for _ in vectors],
    }

    for start in range(0, len(data), 4):
        rows = data[start : start + 4]
        base, source, _, correct, incorrect = prepare_batch(rows, tokenizer, device)
        base_positions = (base["attention_mask"].sum(1) - 1).tolist()
        source_positions = (source["attention_mask"].sum(1) - 1).tolist()
        source_values, source_logits = capture_head(model, source, source_positions)
        with torch.inference_mode():
            clean_logits = model(**base).logits
        full_mean_logits = run_head_intervention(
            model, base, base_positions, mean, "full"
        )
        full_source_logits = run_head_intervention(
            model, base, base_positions, source_values, "full"
        )

        raw["clean"].extend(
            gap_values(clean_logits, base["attention_mask"], correct, incorrect).cpu().tolist()
        )
        raw["source"].extend(
            gap_values(source_logits, source["attention_mask"], correct, incorrect).cpu().tolist()
        )
        raw["full_mean"].extend(
            gap_values(full_mean_logits, base["attention_mask"], correct, incorrect).cpu().tolist()
        )
        raw["full_source"].extend(
            gap_values(full_source_logits, base["attention_mask"], correct, incorrect).cpu().tolist()
        )

        for index, vector in enumerate(vectors):
            for mode, target, key in (
                ("one_d", mean, "one_d_mean"),
                ("one_d", source_values, "one_d_source"),
                ("complement", mean, "complement_mean"),
                ("complement", source_values, "complement_source"),
            ):
                logits = run_head_intervention(
                    model, base, base_positions, target, mode, vector
                )
                raw[key][index].extend(
                    gap_values(
                        logits, base["attention_mask"], correct, incorrect
                    ).cpu().tolist()
                )
        print(f"Head channel {start + len(rows)}/{len(data)}", flush=True)

    return {key: np.asarray(value, dtype=np.float32) for key, value in raw.items()}


def summarize(raw):
    clean = float(raw["clean"].mean())
    source = float(raw["source"].mean())
    denominator = clean - source
    result = {
        "clean_gap": clean,
        "source_gap": source,
        "full": {
            "ablation_gap": float(raw["full_mean"].mean()),
            "intervention_gap": float(raw["full_source"].mean()),
        },
    }
    result["full"]["necessity"] = (
        clean - result["full"]["ablation_gap"]
    ) / clean
    result["full"]["sufficiency"] = (
        clean - result["full"]["intervention_gap"]
    ) / denominator

    for name, mean_key, source_key in (
        ("one_d", "one_d_mean", "one_d_source"),
        ("complement", "complement_mean", "complement_source"),
    ):
        ablation_gaps = raw[mean_key].mean(1)
        intervention_gaps = raw[source_key].mean(1)
        necessity = (clean - ablation_gaps) / clean
        sufficiency = (clean - intervention_gaps) / denominator
        result[name] = {
            "ablation_gap": ablation_gaps.tolist(),
            "intervention_gap": intervention_gaps.tolist(),
            "necessity": necessity.tolist(),
            "sufficiency": sufficiency.tolist(),
            "necessity_mean": float(necessity.mean()),
            "necessity_se": float(necessity.std(ddof=1) / np.sqrt(len(necessity))),
            "sufficiency_mean": float(sufficiency.mean()),
            "sufficiency_se": float(sufficiency.std(ddof=1) / np.sqrt(len(sufficiency))),
        }
    return result


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
    model_name = args.model.split("/")[-1]
    vector_dir = ROOT / "results/cache/das" / model_name / f"step{args.checkpoint}"
    vector_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(3):
        vector = train_direction(model, tokenizer, train_data, args.device, seed)
        torch.save(vector, vector_dir / f"seed{seed}.pt")

    vectors = [
        torch.load(vector_dir / f"seed{seed}.pt", map_location=args.device)
        for seed in range(3)
    ]
    mean = mean_head(model, tokenizer, train_data, args.device)
    raw = evaluate(model, tokenizer, eval_data, mean, vectors, args.device)
    summary = summarize(raw)
    summary.update(
        {"model": args.model, "checkpoint": args.checkpoint, "n": len(eval_data)}
    )

    output_dir = ROOT / "results/experiment2_head_channel" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / f"step{args.checkpoint}.npz", **raw)
    output = output_dir / f"step{args.checkpoint}.json"
    output.write_text(json.dumps(summary, indent=2))
    print(
        f"full mean={summary['full']['ablation_gap']:+.4f} "
        f"necessity={summary['full']['necessity']:.1%} "
        f"source={summary['full']['intervention_gap']:+.4f} "
        f"sufficiency={summary['full']['sufficiency']:.1%}"
    )
    print(
        f"1D mean={np.mean(summary['one_d']['ablation_gap']):+.4f} "
        f"necessity={summary['one_d']['necessity_mean']:.1%} "
        f"source={np.mean(summary['one_d']['intervention_gap']):+.4f} "
        f"sufficiency={summary['one_d']['sufficiency_mean']:.1%}"
    )
    print(output)


if __name__ == "__main__":
    main()
