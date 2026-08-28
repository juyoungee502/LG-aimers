"""Paired TabM screen for training-only pitch-command auxiliary targets.

Both members use exactly the same inference features, architecture,
initialization, and minibatch order.  The reference member learns only the
main Brier objective; the paired member additionally predicts reconstructed
failure details, command class, and pitch group.  Only the paired prediction
difference is evaluated later, isolating representation value from the weak
standalone neural predictor.

Current-pitch labels are reconstructed from training history and are never
inputs.  Trackman context is restricted to 2019--2024 and aggregated strictly
before the row season, so inference remains row-independent.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas.errors import PerformanceWarning
from rtdl_num_embeddings import LinearReLUEmbeddings
from tabm import TabM
from torch import nn

from failure_context import prior_season_context
from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import PITCH_TYPES, bss, reconstruct_labels
from trackman_context import attach_context, pitcher_mapping, prepare_trackman


ROOT = Path(__file__).resolve().parent
CAT_COLUMNS = (
    "base_state", "pitcher_team_id", "batter_team_id", "game_dayofweek",
    "pitcher_hand", "batter_hand", "count_state", "base_out_state",
    "hand_matchup", "game_type", "top_bottom",
)
DETAIL_COLUMNS = ("reverse", "middle", "ball", "strike")
D_OUT = 1 + len(DETAIL_COLUMNS) + 5 + len(PITCH_TYPES)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=6600)
    parser.add_argument("--aux-weight", type=float, default=.08)
    parser.add_argument("--half-life", type=float, default=2.)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--d-block", type=int, default=192)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=8)
    return parser.parse_args()


def categorical_matrix(features: pd.DataFrame, fit: np.ndarray):
    columns, cardinalities = [], []
    for column in CAT_COLUMNS:
        values = pd.unique(features.loc[fit, column].fillna(-1))
        codes = pd.Categorical(
            features[column].fillna(-1), categories=values,
        ).codes.astype(np.int64) + 1
        columns.append(codes)
        cardinalities.append(len(values) + 1)
    return np.column_stack(columns), cardinalities


def numeric_matrix(features: pd.DataFrame, fit: np.ndarray):
    columns = [
        column for column in features.columns
        if column not in CAT_COLUMNS and column not in ("season", "game_month")
    ]
    values = features[columns].to_numpy(np.float32)
    mean = np.nanmean(values[fit], axis=0).astype(np.float32)
    std = np.nanstd(values[fit], axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-5), std, 1.).astype(np.float32)
    return values, columns, mean, std


def numeric_batch(values, indices, mean, std):
    output = (values[indices] - mean) / std
    return np.nan_to_num(
        output, nan=0., posinf=8., neginf=-8.,
    ).clip(-8., 8.).astype(np.float32, copy=False)


def make_model(args, numeric_count, cardinalities, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return TabM.make(
        n_num_features=numeric_count,
        cat_cardinalities=cardinalities,
        d_out=D_OUT,
        num_embeddings=LinearReLUEmbeddings(
            numeric_count, d_embedding=args.embedding_dim,
        ),
        k=args.k,
        d_block=args.d_block,
        n_blocks=args.n_blocks,
        dropout=.10,
        arch_type="tabm",
    ).to(device)


def weighted_mean(loss, weight):
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def batch_loss(logits, target, detail, command, pitch, sample_weight, aux_weight):
    # logits: [batch, ensemble member, output]
    main_probability = torch.sigmoid(logits[:, :, 0])
    main_weight = sample_weight[:, None].expand_as(main_probability)
    main_loss = weighted_mean(
        (main_probability - target[:, None]).square(), main_weight,
    )
    if aux_weight <= 0.:
        return main_loss, (main_loss.detach(), None)

    detail_logits = logits[:, :, 1:1 + len(DETAIL_COLUMNS)]
    detail_target = detail[:, None, :].expand_as(detail_logits)
    detail_mask = torch.isfinite(detail_target)
    detail_weight = (
        sample_weight[:, None, None].expand_as(detail_logits) * detail_mask
    )
    detail_loss = weighted_mean(
        nn.functional.binary_cross_entropy_with_logits(
            detail_logits, torch.nan_to_num(detail_target), reduction="none",
        ),
        detail_weight,
    )

    command_start = 1 + len(DETAIL_COLUMNS)
    command_logits = logits[:, :, command_start:command_start + 5]
    command_mask = command >= 0
    if command_mask.any():
        command_selected = command_logits[command_mask]
        command_target = command[command_mask, None].expand(
            -1, command_selected.shape[1],
        )
        command_raw = nn.functional.cross_entropy(
            command_selected.reshape(-1, 5), command_target.reshape(-1),
            reduction="none",
        ).reshape_as(command_target)
        command_weight = sample_weight[command_mask, None].expand_as(command_raw)
        command_loss = weighted_mean(command_raw, command_weight)
    else:
        command_loss = main_loss.new_zeros(())

    pitch_logits = logits[:, :, command_start + 5:]
    pitch_mask = pitch >= 0
    if pitch_mask.any():
        pitch_selected = pitch_logits[pitch_mask]
        pitch_target = pitch[pitch_mask, None].expand(-1, pitch_selected.shape[1])
        pitch_raw = nn.functional.cross_entropy(
            pitch_selected.reshape(-1, len(PITCH_TYPES)), pitch_target.reshape(-1),
            reduction="none",
        ).reshape_as(pitch_target)
        pitch_weight = sample_weight[pitch_mask, None].expand_as(pitch_raw)
        pitch_loss = weighted_mean(pitch_raw, pitch_weight)
    else:
        pitch_loss = main_loss.new_zeros(())

    auxiliary = .25 * detail_loss + .50 * command_loss + .25 * pitch_loss
    return main_loss + aux_weight * auxiliary, (
        main_loss.detach(), auxiliary.detach(),
    )


@torch.inference_mode()
def predict(model, numeric, categorical, indices, mean, std, batch_size, device):
    model.eval()
    output = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        x_num = torch.from_numpy(numeric_batch(numeric, selected, mean, std)).to(device)
        x_cat = torch.from_numpy(categorical[selected]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(x_num, x_cat)[:, :, 0]
        output[start:start + len(selected)] = (
            torch.sigmoid(logits.float()).mean(dim=1).cpu().numpy()
        )
    return output


def train_variant(
    name, aux_weight, args, numeric, categorical, cardinalities, mean, std,
    target, detail, command, pitch, train_indices, valid_indices, sample_weight,
    device,
):
    model = make_model(args, numeric.shape[1], cardinalities, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    rng = np.random.default_rng(args.seed)
    predictions = {}
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_indices)
        total_loss = total_main = total_aux = 0.
        seen = 0
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            x_num = torch.from_numpy(numeric_batch(numeric, selected, mean, std)).to(device)
            x_cat = torch.from_numpy(categorical[selected]).to(device)
            y = torch.from_numpy(target[selected]).to(device)
            detail_y = torch.from_numpy(detail[selected]).to(device)
            command_y = torch.from_numpy(command[selected]).to(device)
            pitch_y = torch.from_numpy(pitch[selected]).to(device)
            weight = torch.from_numpy(sample_weight[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(x_num, x_cat)
                loss, pieces = batch_loss(
                    logits, y, detail_y, command_y, pitch_y, weight, aux_weight,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            batch_n = len(selected)
            total_loss += float(loss.detach()) * batch_n
            total_main += float(pieces[0]) * batch_n
            if pieces[1] is not None:
                total_aux += float(pieces[1]) * batch_n
            seen += batch_n
        scheduler.step()
        prediction = predict(
            model, numeric, categorical, valid_indices, mean, std,
            args.eval_batch_size, device,
        )
        predictions[f"{name}_epoch_{epoch}"] = prediction
        record = {
            "variant": name, "epoch": epoch,
            "train_loss": total_loss / seen,
            "train_main_brier": total_main / seen,
            "train_auxiliary": total_aux / seen if aux_weight > 0. else None,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_prediction_mean": float(prediction.mean()),
        }
        logs.append(record)
        print(json.dumps(record), flush=True)
    del model, optimizer
    torch.cuda.empty_cache()
    return predictions, logs


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    if args.aux_weight <= 0.:
        raise ValueError("--aux-weight must be positive for the paired screen")
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
        ],
        encoding="utf-8-sig", low_memory=False,
    )
    if int(trackman["season"].max()) > 2024:
        raise ValueError("Forbidden post-2024 Trackman rows detected")
    mapping, mapping_report = pitcher_mapping(ROOT, data, trackman)
    trackman_features = attach_context(data, prepare_trackman(trackman, mapping))

    bases = training_history_arrays(data, target_series)
    features = engineer_features(
        data, *bases, global_prior=float(target.mean()),
    )
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([
        features,
        prior_season_context(full, labels),
        trackman_features,
    ], axis=1)
    features = features.drop(columns=[
        column for column in ("pitcher_id", "batter_id", "team_matchup")
        if column in features
    ])

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

    seasons = data["season"].to_numpy(np.int16)
    train = seasons < args.valid_year
    valid = seasons == args.valid_year
    train_indices = np.flatnonzero(train)
    valid_indices = np.flatnonzero(valid)
    age = (args.valid_year - 1) - seasons[train_indices].astype(float)
    sample_weight = np.exp(
        -math.log(2.) * age / args.half_life,
    ).astype(np.float32)
    full_sample_weight = np.zeros(len(data), dtype=np.float32)
    full_sample_weight[train_indices] = sample_weight

    categorical, cardinalities = categorical_matrix(features, train)
    numeric, numeric_columns, mean, std = numeric_matrix(features, train)
    print(json.dumps({
        "valid_year": args.valid_year,
        "train_rows": len(train_indices),
        "valid_rows": len(valid_indices),
        "numeric_features": len(numeric_columns),
        "categorical_cardinalities": dict(zip(CAT_COLUMNS, cardinalities)),
        "detail_coverage": float(np.isfinite(detail).all(axis=1).mean()),
        "command_coverage": float((command >= 0).mean()),
        "pitch_coverage": float((pitch >= 0).mean()),
        "mapped_pitchers": len(mapping),
        "minimum_mapping_confidence": float(mapping_report["confidence"].min()),
        "forbidden_2025_trackman_used": False,
    }), flush=True)

    all_predictions = {}
    all_logs = []
    for name, aux_weight in (("main", 0.), ("aux", args.aux_weight)):
        predictions, logs = train_variant(
            name, aux_weight, args, numeric, categorical, cardinalities,
            mean, std, target, detail, command, pitch, train_indices,
            valid_indices, full_sample_weight, device,
        )
        all_predictions.update(predictions)
        all_logs.extend(logs)

    valid_target = target[valid_indices].astype(float)
    standalone = {
        name: float(bss(valid_target, prediction.astype(float)))
        for name, prediction in all_predictions.items()
    }
    output = ROOT / "research" / f"v66_multitask_tabm_{args.valid_year}.npz"
    np.savez_compressed(
        output,
        target=valid_target.astype(np.float32),
        logs_json=np.asarray(json.dumps(all_logs)),
        standalone_json=np.asarray(json.dumps(standalone)),
        numeric_columns=np.asarray(numeric_columns),
        cat_columns=np.asarray(CAT_COLUMNS),
        aux_weight=np.asarray(args.aux_weight),
        **{key: value.astype(np.float32) for key, value in all_predictions.items()},
    )
    print(json.dumps({"standalone": standalone, "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
