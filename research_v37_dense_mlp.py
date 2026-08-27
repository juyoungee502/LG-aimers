"""Train a dense numeric MLP as a non-tree ensemble axis.

This deliberately avoids player embeddings and raw player IDs.  Continuous
and low-cardinality pre-pitch features are median-imputed, standardized from
the training fold only, and augmented with missingness indicators.  The model
uses BCE training and is assessed only through chronological OOF blending.
"""
from __future__ import annotations

import argparse
import json
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


ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, default=2024, choices=(2023, 2024))
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--half-life", type=float, default=4.)
    return parser.parse_args()


class DenseMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(.55),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(.55),
            nn.Linear(64, 1),
        )

    def forward(self, values):
        return self.layers(values).squeeze(1)


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def time_masks(length):
    position = np.arange(length)
    result = {
        "all": np.ones(length, dtype=bool),
        "h1": position < length // 2,
        "h2": position >= length // 2,
    }
    for index, part in enumerate(np.array_split(position, 4), 1):
        mask = np.zeros(length, dtype=bool)
        mask[part] = True
        result[f"q{index}"] = mask
    return result


def preprocess(features, train):
    values = features.to_numpy(np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    missing = np.isnan(values)
    median = np.nanmedian(values[train], axis=0).astype(np.float32)
    median = np.nan_to_num(median, nan=0.)
    values = np.where(missing, median[None, :], values).astype(np.float32)
    mean = values[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values[train].std(axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.
    values = np.clip((values - mean) / std, -8., 8.).astype(np.float32)
    missing_columns = missing[train].any(axis=0)
    if missing_columns.any():
        values = np.hstack([
            values, missing[:, missing_columns].astype(np.float32),
        ])
    return values, {
        "median": median,
        "mean": mean,
        "std": std,
        "missing_columns": missing_columns,
    }


@torch.inference_mode()
def predict_logits(model, values, indices, batch_size):
    model.eval()
    output = []
    for start in range(0, len(indices), batch_size * 8):
        chosen = indices[start:start + batch_size * 8]
        output.append(model(values[chosen]).float().cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def train_member(values, target, weights, train_indices, valid_indices, args, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = DenseMLP(values.shape[1]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=.004, weight_decay=.005,
    )
    steps_per_epoch = (len(train_indices) + args.batch_size - 1) // args.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=.004, total_steps=args.epochs * steps_per_epoch,
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_indices), generator=generator, device="cuda")
        total_loss = 0.
        total_weight = 0.
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            if len(selected) < 2:
                continue
            indices = train_indices[selected]
            optimizer.zero_grad(set_to_none=True)
            logits = model(values[indices])
            batch_weights = weights[selected]
            loss = (
                loss_function(logits, target[indices]) * batch_weights
            ).sum() / batch_weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * float(batch_weights.sum())
            total_weight += float(batch_weights.sum())
        print(
            f"v37 seed={seed} epoch={epoch + 1}/{args.epochs} "
            f"weighted_bce={total_loss / total_weight:.7f}", flush=True,
        )
    result = predict_logits(model, values, valid_indices, args.batch_size)
    del model
    torch.cuda.empty_cache()
    return result


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    torch.set_float32_matmul_precision("high")
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    # Raw player numbers encourage ordinal memorisation and do not extrapolate
    # to new 2025 entities.  Their leakage-safe as-of summaries remain.
    features = features.drop(columns=[
        column for column in ("pitcher_id", "batter_id", "team_matchup")
        if column in features
    ])
    seasons = raw["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    matrix, preprocessing = preprocess(features, train)

    device_values = torch.from_numpy(matrix).cuda()
    device_target = torch.from_numpy(target).cuda()
    train_indices = torch.from_numpy(np.flatnonzero(train)).cuda()
    valid_indices = torch.from_numpy(np.flatnonzero(valid)).cuda()
    age = (args.valid_year - 1) - seasons[train].astype(float)
    sample_weight = np.exp(-np.log(2.) * age / args.half_life).astype(np.float32)
    device_weight = torch.from_numpy(sample_weight).cuda()
    logits = []
    for seed_index in range(args.n_seeds):
        logits.append(train_member(
            device_values, device_target, device_weight,
            train_indices, valid_indices, args, 3700 + 1000 * seed_index,
        ))
    prediction = sigmoid(np.mean(logits, axis=0))

    bases_oof = {}
    for version in (23, 24):
        with np.load(ROOT / f"outputs/v{version}_oof_predictions.npz") as archive:
            fold = archive["season"] == args.valid_year
            if not np.allclose(archive["target"][fold], target[valid]):
                raise ValueError(f"v{version} and train.csv rows differ")
            bases_oof[f"v{version}"] = np.clip(
                archive["blended"][fold].astype(float), .005, .995,
            )
    fold_target = target[valid].astype(float)
    game_type = raw.loc[valid, "game_type"].astype(str).to_numpy()
    masks = time_masks(len(fold_target))
    reports = []
    for base_name, base in bases_oof.items():
        direction = logit(prediction) - logit(base)
        for gate in ("all", "R", "F"):
            active = np.ones(len(base), dtype=bool) if gate == "all" else game_type == gate
            for weight in np.round(np.arange(0., .4001, .025), 4):
                candidate = base.copy()
                candidate[active] = sigmoid(logit(base[active]) + weight * direction[active])
                gains = {
                    name: float(
                        bss(fold_target[mask], candidate[mask])
                        - bss(fold_target[mask], base[mask])
                    ) for name, mask in masks.items()
                }
                reports.append({
                    "base": base_name,
                    "gate": gate,
                    "weight": float(weight),
                    "gains": gains,
                    "score": float(bss(fold_target, candidate)),
                    "min_half": float(min(gains["h1"], gains["h2"])),
                    "min_quarter": float(min(gains[f"q{i}"] for i in range(1, 5))),
                })
    reports.sort(
        key=lambda row: (row["min_half"], row["gains"]["all"]), reverse=True,
    )
    diagnostics = {
        "valid_year": args.valid_year,
        "n_seeds": args.n_seeds,
        "epochs": args.epochs,
        "half_life": args.half_life,
        "features_before_missing_indicators": int(features.shape[1]),
        "input_size": int(matrix.shape[1]),
        "missing_indicators": int(preprocessing["missing_columns"].sum()),
        "standalone_bss": float(bss(fold_target, prediction)),
        "correlation_v23": float(np.corrcoef(prediction, bases_oof["v23"])[0, 1]),
        "top": reports[:60],
    }
    output = ROOT / "research" / (
        f"v37_dense_mlp_hl{args.half_life:g}_s{args.n_seeds}_{args.valid_year}.npz"
    )
    np.savez_compressed(
        output,
        target=fold_target.astype(np.float32),
        prediction=prediction.astype(np.float32),
        game_type=np.asarray(game_type, dtype="<U1"),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
