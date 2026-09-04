import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import gap_values, load_data, prepare_batch
from src.relp import score_layer


PATHS = {
    "L15H7": [(15, 7, "subject")],
    "L13H7": [(13, 7, "subject"), (13, 7, "prep")],
}

HEAD_EDGES = {
    "L15H7_subject": [(15, 7, "subject")],
    "L13H7_subject": [(13, 7, "subject")],
    "L13H7_prep": [(13, 7, "prep")],
    **PATHS,
}


def prep_positions(rows, tokenizer):
    prefixes = [row["base"].rsplit(" the ", 1)[0] for row in rows]
    return [
        len(tokenizer(text, add_special_tokens=False)["input_ids"]) - 1
        for text in prefixes
    ]


def compute_relp(model, tokenizer, data, position, device, batch_size):
    n_layers = model.config.num_hidden_layers
    n_neurons = model.gpt_neox.layers[0].mlp.dense_4h_to_h.in_features
    total = torch.zeros(n_layers, n_neurons, dtype=torch.float64)
    for start in range(0, len(data), batch_size):
        rows = data[start : start + batch_size]
        base, _, subject, correct, incorrect = prepare_batch(rows, tokenizer, device)
        if position == "subject":
            positions = subject
        elif position == "prep":
            positions = prep_positions(rows, tokenizer)
        else:
            positions = (base["attention_mask"].sum(1) - 1).tolist()
        for layer in range(n_layers):
            total[layer] += score_layer(
                model,
                base["input_ids"],
                base["attention_mask"],
                correct,
                incorrect,
                positions,
                layer,
            )
        print(
            f"RelP {position} {start + len(rows)}/{len(data)}", flush=True
        )
    return (total / len(data)).to(torch.float32).numpy()


def load_relp(model, tokenizer, data, model_name, checkpoint, position, device):
    suffix = "" if position == "subject" else f"_{position}"
    path = ROOT / "results/cache/relp" / model_name / f"step{checkpoint}{suffix}.npz"
    if path.exists():
        print(f"Loaded {path}")
        return np.load(path)["scores"]
    scores = compute_relp(model, tokenizer, data, position, device, batch_size=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, scores=scores)
    print(f"Saved {path}")
    return scores


def select_candidates(scores_by_position, top_k):
    rows = []
    for position, scores in scores_by_position.items():
        for layer in range(1, scores.shape[0]):
            for neuron in np.flatnonzero(scores[layer] > 0):
                rows.append(
                    {
                        "position": position,
                        "layer": layer,
                        "neuron": int(neuron),
                        "relp": float(scores[layer, neuron]),
                    }
                )
    rows.sort(key=lambda row: (-row["relp"], row["position"], row["layer"], row["neuron"]))
    return rows[:top_k]


def cache_batches(model, tokenizer, data, candidates, device, batch_size):
    candidate_keys = {
        (row["position"], row["layer"], row["neuron"]) for row in candidates
    }
    candidate_layers = sorted({layer for _, layer, _ in candidate_keys})
    tracked_heads = {(13, 7), (15, 7)}
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cached = []

    for start in range(0, len(data), batch_size):
        rows_data = data[start : start + batch_size]
        base, source, subject, correct, incorrect = prepare_batch(
            rows_data, tokenizer, device
        )
        positions = {
            "subject": subject,
            "prep": prep_positions(rows_data, tokenizer),
            "prediction": (base["attention_mask"].sum(1) - 1).tolist(),
        }
        mlp = {}
        head_values = {}

        def capture(inputs, tag):
            stored_mlp = {}
            stored_head_input = {}
            handles = []
            for layer in candidate_layers:
                def make_mlp_hook(layer=layer):
                    def hook(module, args):
                        stored_mlp[layer] = args[0].detach()
                        return args

                    return hook

                handles.append(
                    model.gpt_neox.layers[layer].mlp.dense_4h_to_h
                    .register_forward_pre_hook(make_mlp_hook())
                )
            for layer, _ in tracked_heads:
                def make_head_hook(layer=layer):
                    def hook(module, args):
                        stored_head_input[layer] = args[0].detach()
                        return args

                    return hook

                handles.append(
                    model.gpt_neox.layers[layer].attention
                    .register_forward_pre_hook(make_head_hook())
                )
            try:
                with torch.inference_mode():
                    output = model(**inputs)
            finally:
                for handle in handles:
                    handle.remove()

            batch_rows = torch.arange(len(rows_data), device=device)
            for position, layer, neuron in candidate_keys:
                pos = torch.as_tensor(positions[position], device=device)
                mlp[(tag, position, layer, neuron)] = (
                    stored_mlp[layer][batch_rows, pos, neuron].clone()
                )
            for layer, head in tracked_heads:
                attention = model.gpt_neox.layers[layer].attention
                weight = attention.query_key_value.weight
                bias = attention.query_key_value.bias
                left = (head * 3 + 2) * head_dim
                right = left + head_dim
                for position in ("subject", "prep"):
                    pos = torch.as_tensor(positions[position], device=device)
                    hidden = stored_head_input[layer][batch_rows, pos]
                    head_values[(tag, layer, head, position)] = (
                        hidden @ weight[left:right].T + bias[left:right]
                    ).detach()
            return output.logits

        base_logits = capture(base, "base")
        source_logits = capture(source, "source")
        cached.append(
            {
                "inputs": base,
                "positions": positions,
                "correct": correct,
                "incorrect": incorrect,
                "mlp": mlp,
                "head_values": head_values,
                "base_gap": gap_values(
                    base_logits, base["attention_mask"], correct, incorrect
                ).cpu().numpy(),
                "source_gap": gap_values(
                    source_logits, source["attention_mask"], correct, incorrect
                ).cpu().numpy(),
            }
        )
    return cached


def add_neuron_patch(handles, model, cached, row, tag, device):
    layer = row["layer"]
    neuron = row["neuron"]
    position = row["position"]
    positions = torch.as_tensor(cached["positions"][position], device=device)
    values = cached["mlp"][(tag, position, layer, neuron)]
    batch_rows = torch.arange(len(positions), device=device)

    def hook(module, args):
        activation = args[0].clone()
        activation[batch_rows, positions, neuron] = values.to(activation.dtype)
        return (activation,)

    handles.append(
        model.gpt_neox.layers[layer].mlp.dense_4h_to_h
        .register_forward_pre_hook(hook)
    )


def add_edge_overrides(handles, model, cached, edges, tag, device):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    by_layer = {}
    for edge in edges:
        by_layer.setdefault(edge[0], []).append(edge)

    for layer, layer_edges in by_layer.items():
        attention = model.gpt_neox.layers[layer].attention
        captured = {}

        def attention_hook(module, args, output, captured=captured):
            captured["weights"] = output[1]
            captured["input"] = args[0]
            return output

        def layer_hook(
            module,
            args,
            output,
            layer=layer,
            layer_edges=tuple(layer_edges),
            attention=attention,
            captured=captured,
        ):
            hidden = output[0].clone()
            weights = captured["weights"]
            attention_input = captured["input"]
            qkv_weight = attention.query_key_value.weight
            qkv_bias = attention.query_key_value.bias
            output_weight = attention.dense.weight
            batch_rows = torch.arange(hidden.shape[0], device=device)
            target = torch.as_tensor(cached["positions"]["prediction"], device=device)
            for _, head, source_position in layer_edges:
                source = torch.as_tensor(
                    cached["positions"][source_position], device=device
                )
                value_left = (head * 3 + 2) * head_dim
                value_right = value_left + head_dim
                output_left = head * head_dim
                output_right = output_left + head_dim
                actual = (
                    attention_input[batch_rows, source]
                    @ qkv_weight[value_left:value_right].T
                    + qkv_bias[value_left:value_right]
                )
                replacement = cached["head_values"][
                    (tag, layer, head, source_position)
                ]
                alpha = weights[batch_rows, head, target, source].unsqueeze(1)
                hidden[batch_rows, target] += (
                    alpha * (replacement - actual)
                ) @ output_weight[:, output_left:output_right].T
            return (hidden,) + output[1:]

        handles.append(attention.register_forward_hook(attention_hook))
        handles.append(
            model.gpt_neox.layers[layer].register_forward_hook(layer_hook)
        )


def run_condition(model, cached_batches, device, neuron=None, neuron_tag=None, edges=None, edge_tag=None):
    all_gaps = []
    for cached in cached_batches:
        handles = []
        try:
            if neuron is not None:
                add_neuron_patch(handles, model, cached, neuron, neuron_tag, device)
            if edges:
                add_edge_overrides(handles, model, cached, edges, edge_tag, device)
            with torch.inference_mode():
                logits = model(**cached["inputs"], output_attentions=True).logits
        finally:
            for handle in handles:
                handle.remove()
        all_gaps.append(
            gap_values(
                logits,
                cached["inputs"]["attention_mask"],
                cached["correct"],
                cached["incorrect"],
            ).cpu().numpy()
        )
    return np.concatenate(all_gaps)


def score_candidates(model, cached, candidates, device):
    base_gap = np.concatenate([batch["base_gap"] for batch in cached])
    source_gap = np.concatenate([batch["source_gap"] for batch in cached])
    base_mean = float(base_gap.mean())
    counterfactual_range = base_mean - float(source_gap.mean())
    head_gaps = {
        name: run_condition(
            model, cached, device, edges=edges, edge_tag="source"
        )
        for name, edges in HEAD_EDGES.items()
    }
    joint_edges = PATHS["L15H7"] + PATHS["L13H7"]
    joint_head_gap = run_condition(
        model, cached, device, edges=joint_edges, edge_tag="source"
    )
    raw = {
        "standalone_gap": np.full((len(candidates), len(base_gap)), np.nan, np.float32),
        "l15_control_gap": np.full((len(candidates), len(base_gap)), np.nan, np.float32),
        "l13_control_gap": np.full((len(candidates), len(base_gap)), np.nan, np.float32),
        "joint_control_gap": np.full((len(candidates), len(base_gap)), np.nan, np.float32),
    }
    rows = []

    for index, candidate in enumerate(candidates):
        position = candidate["position"]
        if position in ("subject", "prep"):
            standalone = run_condition(
                model, cached, device, neuron=candidate, neuron_tag="source"
            )
            raw["standalone_gap"][index] = standalone
            total_range = base_mean - float(standalone.mean())
            route_scores = {"L15H7": 0.0, "L13H7": 0.0}
            edges_by_name = (
                {"L15H7": PATHS["L15H7"], "L13H7": [PATHS["L13H7"][0]]}
                if position == "subject"
                else {"L13H7": [PATHS["L13H7"][1]]}
            )
            controls = {}
            for name, edges in edges_by_name.items():
                controlled = run_condition(
                    model,
                    cached,
                    device,
                    neuron=candidate,
                    neuron_tag="source",
                    edges=edges,
                    edge_tag="base",
                )
                controls[name] = controlled
                route_scores[name] = total_range - (
                    base_mean - float(controlled.mean())
                )
                raw[f"{name[:3].lower()}_control_gap"][index] = controlled
            joint = run_condition(
                model,
                cached,
                device,
                neuron=candidate,
                neuron_tag="source",
                edges=sum(edges_by_name.values(), []),
                edge_tag="base",
            )
            raw["joint_control_gap"][index] = joint
            removed = total_range - (base_mean - float(joint.mean()))
            gate_fraction = removed / total_range if abs(total_range) > 1e-8 else 0.0
            direction = "neuron_to_head"
        else:
            route_scores = {}
            for name, edges in PATHS.items():
                restored = run_condition(
                    model,
                    cached,
                    device,
                    neuron=candidate,
                    neuron_tag="base",
                    edges=edges,
                    edge_tag="source",
                )
                raw[f"{name[:3].lower()}_control_gap"][index] = restored
                route_scores[name] = (
                    base_mean - float(head_gaps[name].mean())
                ) - (base_mean - float(restored.mean()))
            joint = run_condition(
                model,
                cached,
                device,
                neuron=candidate,
                neuron_tag="base",
                edges=joint_edges,
                edge_tag="source",
            )
            raw["joint_control_gap"][index] = joint
            joint_range = base_mean - float(joint_head_gap.mean())
            removed = joint_range - (base_mean - float(joint.mean()))
            gate_fraction = removed / joint_range if abs(joint_range) > 1e-8 else 0.0
            total_range = None
            direction = "head_to_neuron"

        positive = {name: max(0.0, route_scores.get(name, 0.0)) for name in PATHS}
        positive_total = sum(positive.values())
        share_l13 = positive["L13H7"] / positive_total if positive_total else 0.5
        rows.append(
            {
                **candidate,
                "direction": direction,
                "total_range": total_range,
                "route_l15": route_scores.get("L15H7", 0.0),
                "route_l13": route_scores.get("L13H7", 0.0),
                "route_share_l13": share_l13,
                "gate_fraction": gate_fraction,
                "on_route": gate_fraction >= 0.05,
            }
        )
        print(f"Routing {index + 1}/{len(candidates)}", flush=True)

    head_sufficiency = {
        name: (base_mean - float(gaps.mean())) / counterfactual_range
        for name, gaps in head_gaps.items()
    }
    return rows, raw, base_gap, source_gap, head_gaps, joint_head_gap, head_sufficiency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[1000, 2000, 5000, 8000, 18000, 50000, 70000, 143000],
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    data = load_data(ROOT / "data/pp_train.jsonl")
    model_name = args.model.split("/")[-1]
    output_dir = ROOT / "results/experiment2_routing" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "checkpoints": {}}

    for checkpoint in args.checkpoints:
        print(f"\nstep{checkpoint}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=f"step{checkpoint}",
            attn_implementation="eager",
        ).to(args.device).eval()
        scores = {
            position: load_relp(
                model, tokenizer, data, model_name, checkpoint, position, args.device
            )
            for position in ("subject", "prep", "prediction")
        }
        candidates = select_candidates(scores, top_k=32)
        cached = cache_batches(model, tokenizer, data, candidates, args.device, batch_size=4)
        (
            rows,
            raw,
            base_gap,
            source_gap,
            head_gaps,
            joint_head_gap,
            head_sufficiency,
        ) = score_candidates(model, cached, candidates, args.device)

        np.savez(
            output_dir / f"step{checkpoint}.npz",
            base_gap=base_gap,
            source_gap=source_gap,
            l15_head_gap=head_gaps["L15H7"],
            l13_head_gap=head_gaps["L13H7"],
            l13_subject_edge_gap=head_gaps["L13H7_subject"],
            l13_prep_edge_gap=head_gaps["L13H7_prep"],
            joint_head_gap=joint_head_gap,
            **raw,
        )
        payload = {
            "checkpoint": checkpoint,
            "n": len(data),
            "head_sufficiency": head_sufficiency,
            "neurons": rows,
        }
        (output_dir / f"step{checkpoint}.json").write_text(
            json.dumps(payload, indent=2)
        )
        summary["checkpoints"][str(checkpoint)] = {
            "head_sufficiency": head_sufficiency,
            "position_counts": {
                position: sum(row["position"] == position for row in rows)
                for position in ("subject", "prep", "prediction")
            },
        }
        print(
            f"L15H7={head_sufficiency['L15H7']:.1%} "
            f"L13H7={head_sufficiency['L13H7']:.1%}",
            flush=True,
        )
        del model, cached
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
