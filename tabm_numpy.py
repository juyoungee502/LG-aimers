"""Small NumPy inference runtime for the exact TabM architecture used by v67."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_matrices(features: pd.DataFrame, configuration: dict):
    expected = configuration["feature_columns"]
    if list(features.columns) != expected:
        missing = [column for column in expected if column not in features]
        unexpected = [column for column in features if column not in expected]
        raise ValueError(
            f"v67 feature schema mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    numeric = features[configuration["numeric_columns"]].to_numpy(np.float32)
    mean = np.asarray(configuration["numeric_mean"], dtype=np.float32)
    std = np.asarray(configuration["numeric_std"], dtype=np.float32)
    numeric = (numeric - mean) / std
    numeric = np.nan_to_num(
        numeric, nan=0.0, posinf=8.0, neginf=-8.0,
    ).clip(-8.0, 8.0).astype(np.float32, copy=False)

    categorical = []
    for column, categories in zip(
        configuration["categorical_columns"],
        configuration["categorical_categories"],
    ):
        lookup = {value: index + 1 for index, value in enumerate(categories)}
        values = features[column].fillna(-1).astype(str)
        categorical.append(
            values.map(lookup).fillna(0).to_numpy(np.int64)
        )
    return numeric, np.column_stack(categorical)


def _forward(weights, numeric, categorical, cardinalities):
    numeric = np.maximum(
        numeric[:, :, None] * weights["num_module.linear.weight"][None, :, :]
        + weights["num_module.linear.bias"][None, :, :],
        0.0,
    ).reshape(len(numeric), -1)
    one_hot = np.zeros(
        (len(categorical), int(sum(cardinalities))), dtype=np.float32,
    )
    offset = 0
    row = np.arange(len(categorical))
    for index, cardinality in enumerate(cardinalities):
        one_hot[row, offset + categorical[:, index]] = 1.0
        offset += int(cardinality)
    representation = np.concatenate([numeric, one_hot], axis=1)
    k = weights["backbone.blocks.0.0.r"].shape[0]
    hidden = np.broadcast_to(
        representation[:, None, :],
        (len(representation), k, representation.shape[1]),
    )
    block = 0
    while f"backbone.blocks.{block}.0.weight" in weights:
        prefix = f"backbone.blocks.{block}.0"
        hidden = np.einsum(
            "bki,oi->bko",
            hidden * weights[f"{prefix}.r"][None, :, :],
            weights[f"{prefix}.weight"],
            optimize=True,
        )
        hidden = (
            hidden * weights[f"{prefix}.s"][None, :, :]
            + weights[f"{prefix}.bias"][None, :, :]
        )
        hidden = np.maximum(hidden, 0.0)
        block += 1
    logits = np.einsum(
        "bki,kio->bko", hidden, weights["output.weight"], optimize=True,
    ) + weights["output.bias"][None, :, :]
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits[:, :, 0], -30.0, 30.0)))
    return probability.mean(axis=1)


def predict_pair(
    model_dir, features: pd.DataFrame, configuration: dict, batch_size: int = 8192,
):
    numeric, categorical = prepare_matrices(features, configuration)
    cardinalities = configuration["categorical_cardinalities"]
    with np.load(model_dir + "/v67_main.npz") as archive:
        main_weights = {key: archive[key] for key in archive.files}
    with np.load(model_dir + "/v67_aux.npz") as archive:
        aux_weights = {key: archive[key] for key in archive.files}
    main = np.empty(len(features), dtype=np.float32)
    aux = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        stop = min(len(features), start + batch_size)
        main[start:stop] = _forward(
            main_weights, numeric[start:stop], categorical[start:stop], cardinalities,
        )
        aux[start:stop] = _forward(
            aux_weights, numeric[start:stop], categorical[start:stop], cardinalities,
        )
    return main, aux
