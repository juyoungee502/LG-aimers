"""Train full-history paired TabM models and export a dependency-free v67 runtime."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from failure_context import prior_season_context
from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import PITCH_TYPES, reconstruct_labels
from research_v66_multitask_tabm import (
    CAT_COLUMNS, DETAIL_COLUMNS, batch_loss, make_model, numeric_batch,
)
from trackman_context import attach_context, pitcher_mapping, prepare_trackman


ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=6600)
    parser.add_argument("--aux-weight", type=float, default=.08)
    parser.add_argument("--half-life", type=float, default=2.)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--d-block", type=int, default=192)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=8)
    return parser.parse_args()


def export_model(model, path):
    arrays = {
        key: value.detach().float().cpu().numpy().astype(np.float32)
        for key, value in model.state_dict().items()
    }
    np.savez_compressed(path, **arrays)


def train_one(
    name, aux_weight, args, numeric, categorical, cardinalities, mean, std,
    target, detail, command, pitch, sample_weight, device,
):
    model = make_model(args, numeric.shape[1], cardinalities, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(target))
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(indices)
        loss_sum = 0.0
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            x_num = torch.from_numpy(
                numeric_batch(numeric, selected, mean, std)
            ).to(device)
            x_cat = torch.from_numpy(categorical[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(x_num, x_cat)
                loss, _ = batch_loss(
                    logits,
                    torch.from_numpy(target[selected]).to(device),
                    torch.from_numpy(detail[selected]).to(device),
                    torch.from_numpy(command[selected]).to(device),
                    torch.from_numpy(pitch[selected]).to(device),
                    torch.from_numpy(sample_weight[selected]).to(device),
                    aux_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(selected)
        scheduler.step()
        print(json.dumps({
            "variant": name, "epoch": epoch,
            "train_loss": loss_sum / len(indices),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }), flush=True)
    model.eval()
    export_model(model, ROOT / "submit/model" / f"v67_{name}.npz")


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    full = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = full[TARGET_COL].astype(np.float32)
    target = target_series.to_numpy(np.float32)
    data = full.drop(columns=[TARGET_COL])
    labels = reconstruct_labels(full)
    trackman = pd.read_csv(
        ROOT / "data/trackman_history.csv",
        usecols=[
            "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
            "balls_before", "strikes_before", "batter_hand", "rel_speed",
        ], encoding="utf-8-sig", low_memory=False,
    )
    if int(trackman["season"].max()) > 2024:
        raise ValueError("Forbidden post-2024 Trackman rows detected")
    mapping, _ = pitcher_mapping(ROOT, data, trackman)
    trackman_features = attach_context(data, prepare_trackman(trackman, mapping))
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([
        features, prior_season_context(full, labels), trackman_features,
    ], axis=1)
    features = features.drop(columns=[
        column for column in ("pitcher_id", "batter_id", "team_matchup")
        if column in features
    ])

    fit = np.ones(len(features), dtype=bool)
    categorical_parts, cardinalities, categories = [], [], []
    for column in CAT_COLUMNS:
        strings = features[column].fillna(-1).astype(str)
        values = pd.unique(strings).tolist()
        codes = pd.Categorical(strings, categories=values).codes.astype(np.int64) + 1
        categorical_parts.append(codes)
        cardinalities.append(len(values) + 1)
        categories.append(values)
    categorical = np.column_stack(categorical_parts)
    numeric_columns = [
        column for column in features.columns
        if column not in CAT_COLUMNS and column not in ("season", "game_month")
    ]
    numeric = features[numeric_columns].to_numpy(np.float32)
    mean = np.nanmean(numeric[fit], axis=0).astype(np.float32)
    std = np.nanstd(numeric[fit], axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-5), std, 1.).astype(np.float32)

    detail = labels[list(DETAIL_COLUMNS)].to_numpy(np.float32)
    complete = labels[["reverse", "middle"]].notna().all(axis=1).to_numpy()
    reverse = labels["reverse"].fillna(0).eq(1).to_numpy()
    middle = labels["middle"].fillna(0).eq(1).to_numpy()
    command = np.full(len(data), -1, dtype=np.int64)
    command[complete & (target == 1)] = 0
    command[complete & (target == 0) & reverse & ~middle] = 1
    command[complete & (target == 0) & ~reverse & middle] = 2
    command[complete & (target == 0) & reverse & middle] = 3
    command[complete & (target == 0) & ~reverse & ~middle] = 4
    pitch = labels["pitch_type"].map(
        {name: index for index, name in enumerate(PITCH_TYPES)}
    ).fillna(-1).to_numpy(np.int64)
    age = 2024 - data["season"].to_numpy(float)
    sample_weight = np.exp(
        -math.log(2.) * age / args.half_life,
    ).astype(np.float32)

    for name, weight in (("main", 0.), ("aux", args.aux_weight)):
        train_one(
            name, weight, args, numeric, categorical, cardinalities, mean, std,
            target, detail, command, pitch, sample_weight, device,
        )

    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "v54_roster_robust_command" not in metadata.get("model_names", []):
        raise ValueError("v67 requires an existing v54 model bundle")
    metadata["model_names"] = [
        name for name in metadata["model_names"]
        if name != "v60_fraction_confidence"
    ]
    metadata["model_names"].append("v67_multitask_tabm")
    metadata.pop("v60_fraction_confidence", None)
    metadata["version"] = "v67_tabm_conservative"
    metadata["v67_multitask_tabm"] = {
        "feature_columns": list(features.columns),
        "numeric_columns": numeric_columns,
        "numeric_mean": mean.astype(float).tolist(),
        "numeric_std": std.astype(float).tolist(),
        "categorical_columns": list(CAT_COLUMNS),
        "categorical_categories": categories,
        "categorical_cardinalities": cardinalities,
        "threshold": 150.0, "r_weight": -.1, "f_weight": .3,
        "training_max_trackman_season": 2024,
        "row_independent": True,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    print(json.dumps({
        "version": metadata["version"], "rows": len(features),
        "numeric_features": len(numeric_columns),
        "model_dir": str(ROOT / "submit/model"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
