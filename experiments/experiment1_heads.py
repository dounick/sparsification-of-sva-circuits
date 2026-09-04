import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import gap_values, load_data, pair_accuracy, prepare_batch


def capture_heads(model, inputs):
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    rows = torch.arange(inputs["input_ids"].shape[0], device=inputs["input_ids"].device)
    positions = inputs["attention_mask"].sum(1) - 1
    captured = [None] * n_layers

    def make_hook(layer):
        def hook(module, args):
            captured[layer] = args[0][rows, positions].reshape(-1, n_heads, head_dim).detach()
            return args

        return hook

    handles = [
        model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(
            make_hook(layer)
        )
        for layer in range(n_layers)
    ]
    try:
        with torch.inference_mode():
            logits = model(**inputs).logits
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(captured), logits


def build_cache(model, tokenizer, data, device, batch_size):
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    mean_sum = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float64)
    source_activations = []
    clean_gaps = []
    source_gaps = []
    for start in range(0, len(data), batch_size):
        rows = data[start : start + batch_size]
        base, source, _, correct, incorrect = prepare_batch(rows, tokenizer, device)
        base_heads, base_logits = capture_heads(model, base)
        source_heads, source_logits = capture_heads(model, source)
        mean_sum += base_heads.sum(1).cpu().to(torch.float64)
        source_activations.append(source_heads.cpu())
        clean_gaps.append(
            gap_values(base_logits, base["attention_mask"], correct, incorrect).cpu()
        )
        source_gaps.append(
            gap_values(source_logits, source["attention_mask"], correct, incorrect).cpu()
        )

    return {
        "mean_activation": (mean_sum / len(data)).to(torch.float32).numpy(),
        "source_activation": torch.cat(source_activations, dim=1).to(torch.float32).numpy(),
        "clean_gap": torch.cat(clean_gaps).to(torch.float32).numpy(),
        "source_gap": torch.cat(source_gaps).to(torch.float32).numpy(),
    }


def repeat_inputs(inputs, repeats):
    return {
        key: value.repeat((repeats,) + (1,) * (value.ndim - 1))
        for key, value in inputs.items()
    }


def scan_heads(model, tokenizer, data, cache, device, batch_size):
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    mean_ablated_gap = np.empty((n_layers, n_heads, len(data)), dtype=np.float32)
    source_patched_gap = np.empty((n_layers, n_heads, len(data)), dtype=np.float32)

    for start in range(0, len(data), batch_size):
        rows_data = data[start : start + batch_size]
        stop = start + len(rows_data)
        base, _, _, correct, incorrect = prepare_batch(rows_data, tokenizer, device)
        positions = base["attention_mask"].sum(1) - 1
        batch_rows = torch.arange(len(rows_data), device=device)

        for layer in range(n_layers):
            expanded = repeat_inputs(base, 2 * n_heads)
            mean = torch.from_numpy(cache["mean_activation"][layer]).to(device)
            source = torch.from_numpy(
                cache["source_activation"][layer, start:stop]
            ).to(device)

            def hook(module, args):
                activation = args[0].clone()
                for head in range(n_heads):
                    left = head * head_dim
                    right = left + head_dim
                    mean_rows = head * len(rows_data) + batch_rows
                    source_rows = (n_heads + head) * len(rows_data) + batch_rows
                    activation[mean_rows, positions, left:right] = mean[head]
                    activation[source_rows, positions, left:right] = source[:, head]
                return (activation,)

            handle = model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(hook)
            try:
                with torch.inference_mode():
                    logits = model(**expanded).logits
            finally:
                handle.remove()

            gaps = gap_values(
                logits,
                expanded["attention_mask"],
                correct.repeat(2 * n_heads),
                incorrect.repeat(2 * n_heads),
            ).reshape(2 * n_heads, len(rows_data))
            mean_ablated_gap[layer, :, start:stop] = gaps[:n_heads].cpu().numpy()
            source_patched_gap[layer, :, start:stop] = gaps[n_heads:].cpu().numpy()

        print(f"Heads {stop}/{len(data)}", flush=True)

    clean_mean = cache["clean_gap"].mean(dtype=np.float64)
    source_mean = cache["source_gap"].mean(dtype=np.float64)
    mean_ablated_mean = mean_ablated_gap.mean(2, dtype=np.float64)
    source_patched_mean = source_patched_gap.mean(2, dtype=np.float64)
    if clean_mean > 0 and clean_mean - source_mean > 0:
        necessity = (clean_mean - mean_ablated_mean) / clean_mean
        sufficiency = (clean_mean - source_patched_mean) / (clean_mean - source_mean)
    else:
        necessity = np.full((n_layers, n_heads), np.nan)
        sufficiency = np.full((n_layers, n_heads), np.nan)
    return mean_ablated_gap, source_patched_gap, necessity, sufficiency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument("--checkpoint", type=int, required=True)
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
    batch_size = 4
    model_name = args.model.split("/")[-1]
    cache_path = ROOT / "results/cache/heads" / model_name / f"step{args.checkpoint}.npz"

    if cache_path.exists():
        with np.load(cache_path) as saved:
            cache = {key: saved[key] for key in saved.files}
        print(f"Loaded {cache_path}")
    else:
        cache = build_cache(model, tokenizer, train_data, args.device, batch_size)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **cache)
        print(f"Saved {cache_path}")

    mean_ablated_gap, source_patched_gap, necessity, sufficiency = scan_heads(
        model, tokenizer, train_data, cache, args.device, batch_size
    )
    joint = np.minimum(necessity, sufficiency)
    mean_ablated_mean = mean_ablated_gap.mean(2, dtype=np.float64)
    source_patched_mean = source_patched_gap.mean(2, dtype=np.float64)
    accuracy, n_pairs = pair_accuracy(cache["clean_gap"], batch_size)

    output = ROOT / "results/experiment1_heads" / model_name / f"step{args.checkpoint}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        clean_gap=cache["clean_gap"],
        source_gap=cache["source_gap"],
        mean_ablated_gap=mean_ablated_gap,
        source_patched_gap=source_patched_gap,
        mean_ablated_gap_mean=mean_ablated_mean,
        source_patched_gap_mean=source_patched_mean,
        necessity=necessity,
        sufficiency=sufficiency,
        joint=joint,
        accuracy=accuracy,
        n_pairs=n_pairs,
    )

    print(
        f"clean_gap={cache['clean_gap'].mean():+.4f} "
        f"source_gap={cache['source_gap'].mean():+.4f} accuracy={accuracy:.4f}"
    )
    flat_joint = joint.reshape(-1)
    order = np.flatnonzero(np.isfinite(flat_joint))
    order = order[np.argsort(-flat_joint[order])][:10]
    for index in order:
        layer = index // model.config.num_attention_heads
        head = index % model.config.num_attention_heads
        print(
            f"L{layer}H{head} mean_ablation_gap={mean_ablated_mean[layer, head]:+.4f} "
            f"necessity={necessity[layer, head]:+.4f} "
            f"source_patch_gap={source_patched_mean[layer, head]:+.4f} "
            f"sufficiency={sufficiency[layer, head]:+.4f}"
        )
    print(output)


if __name__ == "__main__":
    main()
