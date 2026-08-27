"""Chronological TabM screen as a genuinely independent v24 ensemble axis.

TabM (ICLR 2025) trains a parameter-efficient ensemble of MLPs in one network.
Unlike the earlier single embedding MLP experiment, every ensemble member has
its own BatchEnsemble adapters and its loss is optimized separately.  The
script is research-only: it stores epoch predictions and evaluates fixed
chronological folds without changing submission artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rtdl_num_embeddings import LinearReLUEmbeddings
from tabm import TabM
from torch import nn

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss
from research_v23_combined_candidate import logit, sigmoid
from trackman_context import attach_context, pitcher_mapping, prepare_trackman


CAT_COLUMNS = (
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "base_state",
    "base_out_state", "hand_matchup", "team_matchup", "game_type",
    "top_bottom",
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--d-block", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=8)
    return parser.parse_args()


def categorical_matrix(features: pd.DataFrame, train: np.ndarray):
    columns = []
    cardinalities = []
    for column in CAT_COLUMNS:
        values = pd.unique(features.loc[train, column].fillna(-1))
        codes = pd.Categorical(
            features[column].fillna(-1), categories=values,
        ).codes.astype(np.int64) + 1
        columns.append(codes)
        cardinalities.append(len(values) + 1)  # zero is the unknown category
    return np.column_stack(columns), cardinalities


def numeric_matrix(features: pd.DataFrame, train: np.ndarray):
    columns = [
        column for column in features.columns
        if column not in CAT_COLUMNS and column != "game_month"
    ]
    values = features[columns].to_numpy(np.float32)
    fit = values[train]
    mean = np.nanmean(fit, axis=0).astype(np.float32)
    std = np.nanstd(fit, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-5), std, 1.).astype(np.float32)
    del fit
    return values, columns, mean, std


def numeric_batch(values, indices, mean, std):
    output = (values[indices] - mean) / std
    return np.nan_to_num(output, nan=0., posinf=8., neginf=-8.).clip(-8., 8.)


def segment_masks(rows: pd.DataFrame):
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


@torch.inference_mode()
def predict(model, numeric, categorical, indices, mean, std, batch_size, device):
    model.eval()
    output = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        x_num = torch.from_numpy(numeric_batch(numeric, selected, mean, std)).to(device)
        x_cat = torch.from_numpy(categorical[selected]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(x_num, x_cat).squeeze(-1)
        output[start:start + len(selected)] = (
            torch.sigmoid(logits.float()).mean(dim=1).cpu().numpy()
        )
    return output


def evaluate_epoch(target, base, prediction, rows):
    direction = logit(prediction) - logit(base)
    masks = segment_masks(rows)
    curve = []
    for weight in np.arange(-.10, .301, .01):
        candidate = sigmoid(logit(base) + weight * direction)
        gains = {
            name: bss(target[mask], candidate[mask]) - bss(target[mask], base[mask])
            for name, mask in masks.items() if mask.any()
        }
        curve.append({
            "weight": float(weight), "gains": gains,
            "min_half": min(gains["first_half"], gains["second_half"]),
            "min_quarter": min(gains[f"q{i}"] for i in range(1, 5)),
            "min_month": min(
                gains[name] for name in ("months_3_5", "months_6_7", "months_8_11")
            ),
        })
    robust = max(
        curve,
        key=lambda item: (
            item["min_half"], item["min_quarter"], item["gains"]["all"],
        ),
    )
    best_all = max(curve, key=lambda item: item["gains"]["all"])
    return {
        "standalone_bss": bss(target, prediction),
        "prediction_mean": float(prediction.mean()),
        "robust_best": robust, "best_all": best_all,
    }


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    root = Path(__file__).resolve().parent

    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy(np.float32)
    trackman = pd.read_csv(
        root / "data/trackman_history.csv",
        usecols=[
            "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
            "balls_before", "strikes_before", "batter_hand", "rel_speed",
        ], encoding="utf-8-sig", low_memory=False,
    )
    mapping, mapping_report = pitcher_mapping(root, data, trackman)
    context = attach_context(data, prepare_trackman(trackman, mapping))
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([features, context], axis=1)

    seasons = data["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    train_indices = np.flatnonzero(train)
    valid_indices = np.flatnonzero(valid)
    categorical, cardinalities = categorical_matrix(features, train)
    numeric, numeric_columns, mean, std = numeric_matrix(features, train)
    print(json.dumps({
        "train_rows": len(train_indices), "valid_rows": len(valid_indices),
        "numeric_features": len(numeric_columns),
        "cat_cardinalities": dict(zip(CAT_COLUMNS, cardinalities)),
        "mapped_pitchers": len(mapping),
        "minimum_mapping_confidence": float(mapping_report["confidence"].min()),
    }), flush=True)

    model = TabM.make(
        n_num_features=numeric.shape[1], cat_cardinalities=cardinalities,
        d_out=1,
        num_embeddings=LinearReLUEmbeddings(
            numeric.shape[1], d_embedding=args.embedding_dim,
        ),
        k=args.k, d_block=args.d_block, n_blocks=args.n_blocks,
        dropout=.10, arch_type="tabm",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    rng = np.random.default_rng(args.seed)

    with np.load(root / "outputs/v24_oof_predictions.npz") as archive:
        active = archive["season"] == args.valid_year
        y_valid = archive["target"][active].astype(float)
        base = archive["blended"][active].astype(float)
    if not np.allclose(y_valid, target[valid]):
        raise ValueError("v24 OOF rows do not align")
    rows = data.loc[valid].reset_index(drop=True)

    predictions = {}
    reports = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_indices)
        loss_sum = 0.
        seen = 0
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            x_num = torch.from_numpy(
                numeric_batch(numeric, selected, mean, std)
            ).to(device)
            x_cat = torch.from_numpy(categorical[selected]).to(device)
            y = torch.from_numpy(target[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(x_num, x_cat).squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, y[:, None].expand_as(logits), reduction="mean",
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(selected)
            seen += len(selected)
        scheduler.step()
        prediction = predict(
            model, numeric, categorical, valid_indices, mean, std,
            args.eval_batch_size, device,
        )
        report = evaluate_epoch(y_valid, base, prediction, rows)
        report.update({
            "train_logloss": loss_sum / seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        predictions[f"epoch_{epoch}"] = prediction.astype(np.float32)
        reports[str(epoch)] = report
        print(json.dumps({"epoch": epoch, **report}), flush=True)

    output = root / "research" / f"v24_tabm_{args.valid_year}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=y_valid.astype(np.float32), base=base.astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
        numeric_columns=np.asarray(numeric_columns), cat_columns=np.asarray(CAT_COLUMNS),
        **predictions,
    )
    ranking = sorted(
        reports.items(),
        key=lambda item: (
            item[1]["robust_best"]["min_half"],
            item[1]["robust_best"]["min_quarter"],
            item[1]["robust_best"]["gains"]["all"],
        ), reverse=True,
    )
    print(json.dumps({"ranking": ranking}, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
