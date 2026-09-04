import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.circuit import UPSTREAM_NEURONS
from src.data import load_data


def collect(model, tokenizer, words, layers, device, batch_size):
    prefix = tokenizer.encode("<|endoftext|>The", add_special_tokens=False)
    sums = {layer: torch.zeros(model.config.intermediate_size, dtype=torch.float64) for layer in layers}
    squares = {layer: torch.zeros(model.config.intermediate_size, dtype=torch.float64) for layer in layers}

    for start in range(0, len(words), batch_size):
        ids = [tokenizer.encode(" " + word, add_special_tokens=False) for word in words[start:start + batch_size]]
        if any(len(x) != 1 for x in ids):
            raise ValueError("All words must be single tokens")
        input_ids = torch.tensor([prefix + x for x in ids], device=device)
        positions = torch.full((len(ids),), len(prefix), device=device)
        rows = torch.arange(len(ids), device=device)
        captured = {}
        handles = []

        for layer in layers:
            def hook(module, args, layer=layer):
                captured[layer] = args[0][rows, positions].detach().float().cpu()
                return args

            handles.append(model.gpt_neox.layers[layer].mlp.dense_4h_to_h.register_forward_pre_hook(hook))

        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        for handle in handles:
            handle.remove()

        for layer, values in captured.items():
            values = values.to(torch.float64)
            sums[layer] += values.sum(0)
            squares[layer] += values.square().sum(0)

    means = {layer: sums[layer] / len(words) for layer in layers}
    variances = {
        layer: squares[layer] / len(words) - means[layer].square()
        for layer in layers
    }
    return means, variances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument("--checkpoint", type=int, default=143000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    nouns = load_data(ROOT / "data/amplifier_nouns.jsonl")
    non_nouns = load_data(ROOT / "data/amplifier_non_nouns.jsonl")
    words = {
        "singular": [row["singular"] for row in nouns],
        "plural": [row["plural"] for row in nouns],
        "non_noun": [row["word"] for row in non_nouns],
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=f"step{args.checkpoint}"
    ).to(args.device).eval()
    layers = sorted(UPSTREAM_NEURONS)
    stats = {
        name: collect(model, tokenizer, values, layers, args.device, args.batch_size)
        for name, values in words.items()
    }

    rows = []
    for layer, neurons in UPSTREAM_NEURONS.items():
        sg_mean, sg_var = stats["singular"][0][layer], stats["singular"][1][layer]
        pl_mean, pl_var = stats["plural"][0][layer], stats["plural"][1][layer]
        selectivity = (pl_mean - sg_mean) / torch.sqrt((pl_var + sg_var) / 2).clamp_min(1e-6)
        order = torch.argsort(selectivity, descending=True)
        for neuron in neurons:
            rows.append({
                "layer": layer,
                "neuron": neuron,
                "singular_mean": float(sg_mean[neuron]),
                "plural_mean": float(pl_mean[neuron]),
                "non_noun_mean": float(stats["non_noun"][0][layer][neuron]),
                "plural_selectivity_rank": int((order == neuron).nonzero()[0]) + 1,
            })

    result = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "n_singular": len(words["singular"]),
        "n_plural": len(words["plural"]),
        "n_non_noun": len(words["non_noun"]),
        "neurons": rows,
    }
    output = ROOT / "results/experiment2_amplifiers" / args.model.split("/")[-1] / f"step{args.checkpoint}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print("neuron       singular   plural   non-noun   rank")
    for row in rows:
        print(
            f"L{row['layer']}N{row['neuron']:<5} "
            f"{row['singular_mean']:+.3f}    {row['plural_mean']:+.3f}    "
            f"{row['non_noun_mean']:+.3f}       {row['plural_selectivity_rank']}"
        )
    print(output)


if __name__ == "__main__":
    main()
