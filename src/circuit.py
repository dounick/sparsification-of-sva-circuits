import torch

from src.data import prepare_inputs


UPSTREAM_NEURONS = {
    1: [7988],
    5: [3793],
    8: [6308],
    9: [5060, 3101],
    10: [747, 6267],
    11: [6624, 239],
    12: [1535],
}

PRO_READOUT_NEURONS = {14: [1160, 4146, 6888]}
ANTI_READOUT_NEURONS = {14: [6221], 15: [2511, 3382, 2305, 6710]}
TRANSPORT_HEAD = (13, 7)

READOUT_NEURONS = {}
for group in (PRO_READOUT_NEURONS, ANTI_READOUT_NEURONS):
    for layer, neurons in group.items():
        READOUT_NEURONS.setdefault(layer, []).extend(neurons)

SCOPE = {
    **{layer: "subject" for layer in range(1, 13)},
    14: "prediction",
    15: "prediction",
}


def circuit_neurons(layer):
    if layer in UPSTREAM_NEURONS:
        return UPSTREAM_NEURONS[layer]
    return READOUT_NEURONS.get(layer, [])


def capture_scope(model, inputs, subject_positions, prediction_positions):
    stored = {}
    head = {}
    handles = []

    for layer in SCOPE:
        def make_hook(layer=layer):
            def hook(module, args):
                stored[layer] = args[0].detach()
                return args

            return hook

        handles.append(
            model.gpt_neox.layers[layer].mlp.dense_4h_to_h
            .register_forward_pre_hook(make_hook())
        )

    def head_hook(module, args):
        head["value"] = args[0].detach()
        return args

    head_layer, head_index = TRANSPORT_HEAD
    handles.append(
        model.gpt_neox.layers[head_layer].attention.dense
        .register_forward_pre_hook(head_hook)
    )

    try:
        with torch.inference_mode():
            logits = model(**inputs).logits
    finally:
        for handle in handles:
            handle.remove()

    device = inputs["input_ids"].device
    rows = torch.arange(inputs["input_ids"].shape[0], device=device)
    subject = torch.as_tensor(subject_positions, device=device)
    prediction = torch.as_tensor(prediction_positions, device=device)
    activations = {}
    for layer, position in SCOPE.items():
        indices = subject if position == "subject" else prediction
        activations[(layer, position)] = stored[layer][rows, indices].clone()

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    left = head_index * head_dim
    right = left + head_dim
    activations["head"] = head["value"][rows, prediction, left:right].clone()
    return activations, logits


def run_intervention(
    model,
    inputs,
    subject_positions,
    prediction_positions,
    replacements,
    mode="circuit",
    focal=None,
):
    device = inputs["input_ids"].device
    rows = torch.arange(inputs["input_ids"].shape[0], device=device)
    subject = torch.as_tensor(subject_positions, device=device)
    prediction = torch.as_tensor(prediction_positions, device=device)
    handles = []

    layers = SCOPE.items()
    if mode == "circuit":
        layers = [item for item in layers if circuit_neurons(item[0])]

    for layer, position in layers:
        indices = subject if position == "subject" else prediction
        replacement = replacements[(layer, position)].to(device)
        if replacement.ndim == 1:
            replacement = replacement[None, :].expand(len(rows), -1)
        neurons = torch.as_tensor(circuit_neurons(layer), device=device)

        def make_hook(
            indices=indices,
            replacement=replacement,
            neurons=neurons,
            layer=layer,
            position=position,
        ):
            def hook(module, args):
                activation = args[0].clone()
                if mode == "scope":
                    activation[rows, indices] = replacement.to(activation.dtype)
                elif mode == "circuit":
                    if len(neurons):
                        activation[
                            rows[:, None], indices[:, None], neurons[None, :]
                        ] = replacement[:, neurons].to(activation.dtype)
                elif mode == "off_circuit":
                    values = replacement.clone()
                    if len(neurons):
                        values[:, neurons] = focal[(layer, position)][
                            :, neurons
                        ].to(values.device)
                    activation[rows, indices] = values.to(activation.dtype)
                else:
                    raise ValueError(mode)
                return (activation,)

            return hook

        handles.append(
            model.gpt_neox.layers[layer].mlp.dense_4h_to_h
            .register_forward_pre_hook(make_hook())
        )

    if mode != "off_circuit":
        head_layer, head_index = TRANSPORT_HEAD
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        left = head_index * head_dim
        right = left + head_dim
        head_replacement = replacements["head"].to(device)
        if head_replacement.ndim == 1:
            head_replacement = head_replacement[None, :].expand(len(rows), -1)

        def head_hook(module, args):
            activation = args[0].clone()
            activation[rows, prediction, left:right] = head_replacement.to(
                activation.dtype
            )
            return (activation,)

        handles.append(
            model.gpt_neox.layers[head_layer].attention.dense
            .register_forward_pre_hook(head_hook)
        )

    try:
        with torch.inference_mode():
            return model(**inputs).logits
    finally:
        for handle in handles:
            handle.remove()


def mean_scope_activations(model, tokenizer, data, device, batch_size=4):
    sums = {}
    count = 0
    for start in range(0, len(data), batch_size):
        rows = data[start : start + batch_size]
        base, _, subject_positions = prepare_inputs(rows, tokenizer, device)
        prediction_positions = (base["attention_mask"].sum(1) - 1).tolist()
        activations, _ = capture_scope(
            model, base, subject_positions, prediction_positions
        )
        for key, value in activations.items():
            if key not in sums:
                sums[key] = torch.zeros(value.shape[1], dtype=torch.float64)
            sums[key] += value.sum(0).cpu().to(torch.float64)
        count += len(rows)
    return {
        key: (value / count).to(device=device, dtype=torch.float32)
        for key, value in sums.items()
    }
