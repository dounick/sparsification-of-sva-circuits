import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import gap_values, load_data, pair_accuracy, prepare_batch
from src.relp import score_layer


def capture_activations(model, inputs, positions, layers):
    captured = {}
    rows = torch.arange(inputs["input_ids"].shape[0], device=inputs["input_ids"].device)
    pos = torch.tensor(positions, device=inputs["input_ids"].device)

    def make_hook(layer):
        def hook(module, args):
            captured[layer] = args[0][rows, pos].detach()
            return args

        return hook

    handles = [
        model.gpt_neox.layers[layer].mlp.dense_4h_to_h.register_forward_pre_hook(
            make_hook(layer)
        )
        for layer in layers
    ]
    try:
        with torch.inference_mode():
            logits = model(**inputs).logits
    finally:
        for handle in handles:
            handle.remove()
    return captured, logits


def run_intervention(model, inputs, positions, groups, replacements):
    rows = torch.arange(inputs["input_ids"].shape[0], device=inputs["input_ids"].device)
    pos = torch.tensor(positions, device=inputs["input_ids"].device)
    handles = []

    for layer, neurons_numpy in groups.items():
        neurons = torch.tensor(neurons_numpy, device=rows.device)
        replacement = replacements[layer][:, neurons].to(rows.device)

        def make_hook(neurons=neurons, replacement=replacement):
            def hook(module, args):
                activation = args[0].clone()
                activation[rows[:, None], pos[:, None], neurons[None, :]] = replacement.to(
                    activation.dtype
                )
                return (activation,)

            return hook

        handles.append(
            model.gpt_neox.layers[layer].mlp.dense_4h_to_h.register_forward_pre_hook(
                make_hook()
            )
        )

    try:
        with torch.inference_mode():
            return model(**inputs).logits
    finally:
        for handle in handles:
            handle.remove()


def rank_neurons(scores):
    n_layers, n_neurons = scores.shape
    layers = np.repeat(np.arange(1, n_layers), n_neurons)
    neurons = np.tile(np.arange(n_neurons), n_layers - 1)
    values = scores[1:].reshape(-1)
    keep = values > 0
    layers = layers[keep]
    neurons = neurons[keep]
    values = values[keep]
    order = np.lexsort((neurons, layers, -values))
    return layers[order], neurons[order], values[order]


def prefix_schedule(size):
    values = [0]
    k = 1
    while k < size:
        values.append(k)
        k *= 2
    values.append(size)
    return np.array(sorted(set(values)))


def prefix_groups(layers, neurons, k):
    return {
        int(layer): neurons[:k][layers[:k] == layer]
        for layer in np.unique(layers[:k])
    }


def threshold_crossing(k_values, values, threshold):
    for i in range(1, len(k_values)):
        if not np.isfinite(values[i]) or values[i] < threshold:
            continue
        if k_values[i - 1] == 0 or values[i] == values[i - 1]:
            return float(k_values[i])
        weight = (threshold - values[i - 1]) / (values[i] - values[i - 1])
        return float(
            2
            ** (
                math.log2(k_values[i - 1])
                + weight * (math.log2(k_values[i]) - math.log2(k_values[i - 1]))
            )
        )
    return float("nan")


def compute_relp(model, tokenizer, data, device, batch_size):
    n_layers = model.config.num_hidden_layers
    n_neurons = model.gpt_neox.layers[0].mlp.dense_4h_to_h.weight.shape[1]
    sums = torch.zeros(n_layers, n_neurons, dtype=torch.float64)

    for start in range(0, len(data), batch_size):
        rows = data[start : start + batch_size]
        base, _, positions, correct, incorrect = prepare_batch(
            rows, tokenizer, device
        )
        for layer in range(n_layers):
            sums[layer] += score_layer(
                model,
                base["input_ids"],
                base["attention_mask"],
                correct,
                incorrect,
                positions,
                layer,
            )
        if start % (10 * batch_size) == 0:
            print(f"RelP {start + len(rows)}/{len(data)}", flush=True)

    return (sums / len(data)).to(torch.float32).numpy()


def compute_means(model, tokenizer, data, layers, device, batch_size):
    n_neurons = model.gpt_neox.layers[0].mlp.dense_4h_to_h.weight.shape[1]
    sums = {layer: torch.zeros(n_neurons, dtype=torch.float64) for layer in layers}

    for start in range(0, len(data), batch_size):
        rows = data[start : start + batch_size]
        base, _, positions, _, _ = prepare_batch(rows, tokenizer, device)
        activations, _ = capture_activations(model, base, positions, layers)
        for layer in layers:
            sums[layer] += activations[layer].sum(0).cpu().to(torch.float64)

    return {
        layer: (value / len(data)).to(device=device, dtype=torch.float32)[None, :]
        for layer, value in sums.items()
    }


def compute_frontier(
    model,
    tokenizer,
    train_data,
    eval_data,
    candidate_layers,
    candidate_neurons,
    device,
    batch_size,
):
    layers = sorted(set(int(layer) for layer in candidate_layers))
    means = compute_means(model, tokenizer, train_data, layers, device, batch_size)
    k_values = prefix_schedule(len(candidate_layers))
    clean_gap = np.empty(len(eval_data), dtype=np.float32)
    source_gap = np.empty(len(eval_data), dtype=np.float32)
    source_patched_gap = np.empty((len(k_values), len(eval_data)), dtype=np.float32)
    mean_ablated_gap = np.empty((len(k_values), len(eval_data)), dtype=np.float32)

    for start in range(0, len(eval_data), batch_size):
        rows = eval_data[start : start + batch_size]
        stop = start + len(rows)
        base, source, positions, correct, incorrect = prepare_batch(
            rows, tokenizer, device
        )
        source_activations, source_logits = capture_activations(
            model, source, positions, layers
        )
        with torch.inference_mode():
            clean_logits = model(**base).logits
        clean = gap_values(clean_logits, base["attention_mask"], correct, incorrect)
        counterfactual = gap_values(
            source_logits, source["attention_mask"], correct, incorrect
        )
        clean_gap[start:stop] = clean.cpu().numpy()
        source_gap[start:stop] = counterfactual.cpu().numpy()
        source_patched_gap[0, start:stop] = clean_gap[start:stop]
        mean_ablated_gap[0, start:stop] = clean_gap[start:stop]

        batch_means = {
            layer: means[layer].expand(len(rows), -1) for layer in layers
        }
        for i, k in enumerate(k_values[1:], start=1):
            groups = prefix_groups(candidate_layers, candidate_neurons, int(k))
            patched_logits = run_intervention(
                model, base, positions, groups, source_activations
            )
            ablated_logits = run_intervention(
                model, base, positions, groups, batch_means
            )
            source_patched_gap[i, start:stop] = gap_values(
                patched_logits, base["attention_mask"], correct, incorrect
            ).cpu().numpy()
            mean_ablated_gap[i, start:stop] = gap_values(
                ablated_logits, base["attention_mask"], correct, incorrect
            ).cpu().numpy()

        print(f"Frontier {stop}/{len(eval_data)}", flush=True)

    clean_mean = clean_gap.mean(dtype=np.float64)
    source_mean = source_gap.mean(dtype=np.float64)
    source_patched_mean = source_patched_gap.mean(1, dtype=np.float64)
    mean_ablated_mean = mean_ablated_gap.mean(1, dtype=np.float64)
    if clean_mean > 0 and clean_mean - source_mean > 0:
        sufficiency = (clean_mean - source_patched_mean) / (clean_mean - source_mean)
        necessity = (clean_mean - mean_ablated_mean) / clean_mean
    else:
        sufficiency = np.full(len(k_values), np.nan)
        necessity = np.full(len(k_values), np.nan)
    return (
        k_values,
        clean_gap,
        source_gap,
        source_patched_gap,
        mean_ablated_gap,
        source_patched_mean,
        mean_ablated_mean,
        necessity,
        sufficiency,
        pair_accuracy(clean_gap, batch_size),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
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
    model_name = args.model.split("/")[-1]

    cache_path = (
        ROOT
        / "results/cache/relp"
        / model_name
        / f"step{args.checkpoint}.npz"
    )
    if cache_path.exists():
        scores = np.load(cache_path)["scores"]
        print(f"Loaded {cache_path}")
    else:
        scores = compute_relp(model, tokenizer, train_data, args.device, batch_size)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, scores=scores)
        print(f"Saved {cache_path}")

    candidate_layers, candidate_neurons, candidate_scores = rank_neurons(scores)
    (
        k_values,
        clean_gap,
        source_gap,
        source_patched_gap,
        mean_ablated_gap,
        source_patched_mean,
        mean_ablated_mean,
        necessity,
        sufficiency,
        (accuracy, n_pairs),
    ) = compute_frontier(
        model,
        tokenizer,
        train_data,
        eval_data,
        candidate_layers,
        candidate_neurons,
        args.device,
        batch_size,
    )

    joint = np.minimum(necessity, sufficiency)
    k_threshold = threshold_crossing(k_values, joint, args.threshold)
    output = (
        ROOT
        / "results/experiment1_neurons"
        / model_name
        / f"step{args.checkpoint}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        candidate_layer=candidate_layers,
        candidate_neuron=candidate_neurons,
        candidate_score=candidate_scores,
        k=k_values,
        clean_gap=clean_gap,
        source_gap=source_gap,
        source_patched_gap=source_patched_gap,
        mean_ablated_gap=mean_ablated_gap,
        clean_gap_mean=clean_gap.mean(),
        source_gap_mean=source_gap.mean(),
        source_patched_gap_mean=source_patched_mean,
        mean_ablated_gap_mean=mean_ablated_mean,
        necessity=necessity,
        sufficiency=sufficiency,
        joint=joint,
        accuracy=accuracy,
        n_pairs=n_pairs,
        threshold=args.threshold,
        k_threshold=k_threshold,
    )
    print(
        f"clean_gap={clean_gap.mean():+.4f} source_gap={source_gap.mean():+.4f} "
        f"accuracy={accuracy:.4f}"
    )
    for k, ablated, nec, patched, suff in zip(
        k_values, mean_ablated_mean, necessity, source_patched_mean, sufficiency
    ):
        print(
            f"k={k:<6d} mean_ablation_gap={ablated:+.4f} necessity={nec:+.4f} "
            f"source_patch_gap={patched:+.4f} sufficiency={suff:+.4f}"
        )
    print(f"k{round(100 * args.threshold)}={k_threshold:.3f}")
    print(output)


if __name__ == "__main__":
    main()
