"""Chronological embedding MLP as an independent v23 ensemble axis."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss


SEEDS = (6101, 6102)
VARIANTS = {
    "decay2": {"window": None, "half_life": 2.0},
    "recent2": {"window": 2, "half_life": None},
}
CAT_COLUMNS = (
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "base_state",
    "base_out_state", "hand_matchup", "team_matchup", "game_type",
    "top_bottom",
)
BATCH_SIZE = 8192
EPOCHS = 4


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def masks(rows):
    position = np.arange(len(rows))
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": position < len(rows) // 2,
        "second_half": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }


def embedding_dimension(cardinality):
    return min(32, max(3, int(round(3.0 * cardinality ** .25))))


class EmbeddingMLP(nn.Module):
    def __init__(self, numeric_count, cardinalities):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, embedding_dimension(cardinality))
            for cardinality in cardinalities
        ])
        input_count = numeric_count + sum(
            layer.embedding_dim for layer in self.embeddings
        )
        self.input = nn.Sequential(
            nn.Linear(input_count, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(.10),
        )
        self.residual = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(.10),
            nn.Linear(256, 256), nn.Dropout(.05),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, numeric, categorical):
        embedded = [
            layer(categorical[:, index])
            for index, layer in enumerate(self.embeddings)
        ]
        hidden = self.input(torch.cat([numeric, *embedded], dim=1))
        hidden = hidden + self.residual(hidden)
        return self.output(hidden).squeeze(1)


def categorical_matrix(features, fit):
    columns, cardinalities = [], []
    for column in CAT_COLUMNS:
        values = pd.unique(features.loc[fit, column].fillna(-1))
        codes = pd.Categorical(
            features[column].fillna(-1), categories=values,
        ).codes.astype(np.int64) + 1
        columns.append(codes)
        cardinalities.append(len(values))
    return np.column_stack(columns), cardinalities


def numeric_statistics(numeric, fit):
    selected = numeric[fit]
    mean = np.nanmean(selected, axis=0).astype(np.float32)
    std = np.nanstd(selected, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-5), std, 1.).astype(np.float32)
    return mean, std


def numeric_batch(numeric, indices, mean, std):
    values = (numeric[indices] - mean) / std
    return np.nan_to_num(values, nan=0., posinf=8., neginf=-8.).clip(-8., 8.)


@torch.inference_mode()
def predict(model, numeric, categorical, indices, mean, std, device):
    model.eval()
    output = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), BATCH_SIZE * 2):
        batch_indices = indices[start:start + BATCH_SIZE * 2]
        numeric_tensor = torch.from_numpy(
            numeric_batch(numeric, batch_indices, mean, std)
        ).to(device)
        categorical_tensor = torch.from_numpy(
            categorical[batch_indices]
        ).to(device)
        output[start:start + len(batch_indices)] = torch.sigmoid(
            model(numeric_tensor, categorical_tensor)
        ).cpu().numpy()
    return output


def fit_predict(
    numeric, categorical, cardinalities, target, seasons, valid_year,
    variant_name, variant, seed, mean, std, device,
):
    fit = seasons < valid_year
    if variant["window"] is not None:
        fit &= seasons >= valid_year - int(variant["window"])
    train_indices = np.flatnonzero(fit)
    valid_indices = np.flatnonzero(seasons == valid_year)
    sample_weight = np.ones(len(train_indices), dtype=np.float32)
    if variant["half_life"] is not None:
        age = (valid_year - 1) - seasons[train_indices].astype(float)
        sample_weight = np.exp(
            -math.log(2.) * age / float(variant["half_life"])
        ).astype(np.float32)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = EmbeddingMLP(numeric.shape[1], cardinalities).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.2e-3, weight_decay=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS,
    )
    rng = np.random.default_rng(seed)
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(len(train_indices))
        loss_sum = 0.0
        weight_sum = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            selected = order[start:start + BATCH_SIZE]
            batch_indices = train_indices[selected]
            x_numeric = torch.from_numpy(
                numeric_batch(numeric, batch_indices, mean, std)
            ).to(device)
            x_categorical = torch.from_numpy(
                categorical[batch_indices]
            ).to(device)
            y = torch.from_numpy(target[batch_indices].astype(np.float32)).to(device)
            weight = torch.from_numpy(sample_weight[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            probability = torch.sigmoid(model(x_numeric, x_categorical))
            loss = ((probability - y).square() * weight).sum() / weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * float(weight.sum())
            weight_sum += float(weight.sum())
        scheduler.step()
        print(
            f"MLP year={valid_year} variant={variant_name} seed={seed} "
            f"epoch={epoch + 1}/{EPOCHS} brier={loss_sum / weight_sum:.7f}",
            flush=True,
        )
    prediction = predict(
        model, numeric, categorical, valid_indices, mean, std, device,
    )
    del model
    torch.cuda.empty_cache()
    return prediction


def main():
    root = Path(__file__).resolve().parent
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for this research script")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    raw = pd.read_csv(
        root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(
        raw, *bases, global_prior=float(target.mean()),
    )
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    numeric_columns = [
        column for column in features.columns
        if column not in CAT_COLUMNS and column not in ("season", "game_month")
    ]
    numeric = features[numeric_columns].to_numpy(np.float32)
    seasons = raw["season"].to_numpy(np.int16)
    with np.load(root / "outputs/v23_oof_predictions.npz") as source:
        v23 = {key: source[key] for key in source.files}

    predictions = {}
    reports = []
    for valid_year in (2023, 2024):
        strict_train = seasons < valid_year
        categorical, cardinalities = categorical_matrix(features, strict_train)
        mean, std = numeric_statistics(numeric, strict_train)
        members = {}
        for variant_name, variant in VARIANTS.items():
            members[variant_name] = []
            for seed in SEEDS:
                members[variant_name].append(fit_predict(
                    numeric, categorical, cardinalities, target, seasons,
                    valid_year, variant_name, variant, seed, mean, std, device,
                ))
            predictions[(valid_year, variant_name)] = np.mean(
                members[variant_name], axis=0,
            )
        predictions[(valid_year, "mean")] = np.mean([
            predictions[(valid_year, name)] for name in VARIANTS
        ], axis=0)

        fold = v23["season"] == valid_year
        y = v23["target"][fold].astype(float)
        base = v23["blended"][fold].astype(float)
        rows = raw.loc[seasons == valid_year].reset_index(drop=True)
        if not np.allclose(y, target[seasons == valid_year]):
            raise ValueError(f"v23 rows do not align for {valid_year}")
        for name in (*VARIANTS, "mean"):
            candidate = predictions[(valid_year, name)].astype(float)
            direction = logit(candidate) - logit(base)
            for weight in np.arange(-.10, .301, .01):
                blended = sigmoid(logit(base) + weight * direction)
                gains = {
                    label: bss(y[mask], blended[mask]) - bss(y[mask], base[mask])
                    for label, mask in masks(rows).items() if mask.any()
                }
                reports.append({
                    "year": valid_year, "model": name, "weight": float(weight),
                    "gains": gains, "standalone_bss": bss(y, candidate),
                    "standalone_mean": float(candidate.mean()),
                })

    paired = []
    for name in (*VARIANTS, "mean"):
        for weight in np.arange(-.10, .301, .01):
            selected = [
                row for row in reports if row["model"] == name
                and abs(row["weight"] - weight) < 1e-8
            ]
            by_year = {str(row["year"]): row["gains"] for row in selected}
            temporal = [
                value for row in selected for label, value in row["gains"].items()
                if label != "all"
            ]
            paired.append({
                "model": name, "weight": float(weight), "gains": by_year,
                "min_segment": min(temporal),
                "min_year": min(row["gains"]["all"] for row in selected),
                "mean_year": np.mean([row["gains"]["all"] for row in selected]),
                "standalone": {
                    str(row["year"]): {
                        "bss": row["standalone_bss"],
                        "mean": row["standalone_mean"],
                    } for row in selected
                },
            })
    paired.sort(
        key=lambda row: (row["min_segment"], row["min_year"], row["mean_year"]),
        reverse=True,
    )
    output = root / "research/v23_embedding_mlp.npz"
    np.savez_compressed(
        output,
        names=np.asarray(list(VARIANTS) + ["mean"]),
        prediction_2023=np.column_stack([
            predictions[(2023, name)] for name in (*VARIANTS, "mean")
        ]).astype(np.float32),
        prediction_2024=np.column_stack([
            predictions[(2024, name)] for name in (*VARIANTS, "mean")
        ]).astype(np.float32),
        reports_json=np.asarray(json.dumps(paired)),
    )
    print(json.dumps({"top": paired[:60]}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
