import json

import numpy as np
import torch


def load_data(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def prepare_inputs(rows, tokenizer, device):
    base = tokenizer(
        [row["base"] for row in rows],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    source = tokenizer(
        [row["source"] for row in rows],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    positions = [
        int((base["input_ids"][i] != source["input_ids"][i]).nonzero()[0])
        for i in range(len(rows))
    ]
    return base.to(device), source.to(device), positions


def prepare_batch(rows, tokenizer, device):
    base, source, positions = prepare_inputs(rows, tokenizer, device)
    correct = tokenizer(
        [row["base_label"] for row in rows], add_special_tokens=False
    )["input_ids"]
    incorrect = tokenizer(
        [row["source_label"] for row in rows], add_special_tokens=False
    )["input_ids"]
    return (
        base,
        source,
        positions,
        torch.tensor([ids[0] for ids in correct], device=device),
        torch.tensor([ids[0] for ids in incorrect], device=device),
    )


def gap_values(logits, attention_mask, correct, incorrect):
    rows = torch.arange(logits.shape[0], device=logits.device)
    positions = attention_mask.sum(1) - 1
    last_logits = logits[rows, positions]
    return last_logits[rows, correct] - last_logits[rows, incorrect]


def pair_accuracy(gaps, batch_size=4):
    correct = []
    half = batch_size // 2
    for start in range(0, len(gaps), batch_size):
        batch = gaps[start : start + batch_size]
        correct.extend(batch[i] > 0 and batch[i + half] > 0 for i in range(half))
    return float(np.mean(correct)), len(correct)
