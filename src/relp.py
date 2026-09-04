"""Pythia RelP rules adapted from TransluceAI/circuits."""

import contextlib
import math

import torch
import torch.nn.functional as F
import transformers.models.gpt_neox.modeling_gpt_neox as neox_mod


def _make_linearised_ln_forward(ln):
    weight = ln.weight
    bias = ln.bias
    eps = ln.eps

    def forward(x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        scale = torch.rsqrt(var + eps)
        return (x - mean.detach()) * scale.detach() * weight + bias

    return forward


def _linearised_gelu(x):
    gate = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x * gate.detach()


class _LinearisedGelu(torch.nn.Module):
    def forward(self, x):
        return _linearised_gelu(x)


def _relp_eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    head_mask=None,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = attn_weights.detach()
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


@contextlib.contextmanager
def replacement_model(model):
    layer_norms = []
    for layer in model.gpt_neox.layers:
        layer_norms.append(layer.input_layernorm)
        layer_norms.append(layer.post_attention_layernorm)
    layer_norms.append(model.gpt_neox.final_layer_norm)

    original_layer_norm_forwards = {}
    for layer_norm in layer_norms:
        original_layer_norm_forwards[id(layer_norm)] = layer_norm.forward
        layer_norm.forward = _make_linearised_ln_forward(layer_norm)

    original_activations = {}
    for layer in model.gpt_neox.layers:
        original_activations[id(layer.mlp)] = layer.mlp.act
        layer.mlp.act = _LinearisedGelu()

    original_attention_forward = neox_mod.eager_attention_forward
    neox_mod.eager_attention_forward = _relp_eager_attention_forward

    try:
        yield
    finally:
        for layer_norm in layer_norms:
            layer_norm.forward = original_layer_norm_forwards[id(layer_norm)]
        for layer in model.gpt_neox.layers:
            layer.mlp.act = original_activations[id(layer.mlp)]
        neox_mod.eager_attention_forward = original_attention_forward


def score_layer(
    model,
    input_ids,
    attention_mask,
    correct_token_ids,
    incorrect_token_ids,
    capture_positions,
    layer_index,
):
    captured = {}

    def capture_hook(module, args):
        captured["activation"] = args[0]
        return args

    capture_handle = (
        model.gpt_neox.layers[layer_index]
        .mlp.dense_4h_to_h.register_forward_pre_hook(capture_hook)
    )

    def detach_hook(module, args, output):
        return output.detach()

    stop_handles = []
    for other_layer in range(model.config.num_hidden_layers):
        if other_layer != layer_index:
            stop_handles.append(
                model.gpt_neox.layers[other_layer].mlp.register_forward_hook(detach_hook)
            )

    try:
        model.zero_grad(set_to_none=True)
        with replacement_model(model):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        last_positions = attention_mask.sum(1) - 1
        last_logits = logits[rows, last_positions]
        metric = (
            last_logits[rows, correct_token_ids]
            - last_logits[rows, incorrect_token_ids]
        ).sum()

        activation = captured["activation"]
        gradient = torch.autograd.grad(metric, activation)[0]
        positions = torch.as_tensor(capture_positions, device=input_ids.device)
        scores = activation.detach()[rows, positions] * gradient[rows, positions]
        return scores.detach().sum(0).cpu().to(torch.float64)
    finally:
        capture_handle.remove()
        for handle in stop_handles:
            handle.remove()
