"""Build a non-redundant portfolio from the v24 exhaustive transfer screen."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss
from research_v24_exhaustive_transfer import (
    encode_categorical, encode_numeric, table_direction,
)


ROOT = Path(__file__).resolve().parent
MAX_STEPS = 8


def correction_gain(target, base, correction):
    prediction = np.clip(base + correction, .005, .995)
    return bss(target, prediction) - bss(target, base)


def feature_frames(raw, target_series):
    bases = training_history_arrays(raw, target_series)
    features = engineer_features(raw, *bases, global_prior=float(target_series.mean()))
    add_training_component_features(features, raw)
    features = add_state_interactions(features)
    categorical = raw[[
        column for column in raw.columns
        if raw[column].dtype == "object" or column.endswith("_id")
    ]].copy()
    for column in (
        "balls_before", "strikes_before", "outs_before", "inning",
        "pitcher_hand", "batter_hand", "num_runners_on",
    ):
        categorical[column] = raw[column]
    return features, categorical


def encode(spec, source, valid, numeric, categorical):
    values = numeric[spec["column"]] if spec["kind"] == "numeric" \
        else categorical[spec["column"]]
    if spec["kind"] == "numeric":
        return encode_numeric(values.iloc[source], values.iloc[valid], int(spec["bins"]))
    return encode_categorical(values.iloc[source], values.iloc[valid])


def direction(spec, source, valid, numeric, categorical, target, base):
    encoded = encode(spec, source, valid, numeric, categorical)
    if encoded is None:
        raise ValueError(f"Could not encode {spec}")
    return table_direction(
        encoded[0], encoded[1], target[source] - base[source], float(spec["shrink"]),
    ) * float(spec["scale"])


def detail(target, base, candidate, rows):
    position = np.arange(len(rows))
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "half_1": position < len(rows) // 2,
        "half_2": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
        **{
            f"month_{month}": rows["game_month"].eq(month).to_numpy()
            for month in sorted(rows["game_month"].unique())
        },
    }
    return {
        name: bss(target[active], candidate[active]) - bss(target[active], base[active])
        for name, active in masks.items() if active.any()
    }


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    numeric_all, categorical_all = feature_frames(raw, target_series)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    seasons = raw["season"].to_numpy(np.int16)
    positions = np.concatenate([np.flatnonzero(seasons == year) for year in (2023, 2024)])
    if not np.allclose(target_all[positions], oof["target"]):
        raise ValueError("v24 OOF rows do not align")
    numeric = numeric_all.iloc[positions].reset_index(drop=True)
    categorical = categorical_all.iloc[positions].reset_index(drop=True)
    rows = raw.iloc[positions].reset_index(drop=True)
    target = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    year = oof["season"].astype(int)
    regular = rows["game_type"].eq("R").to_numpy()
    regular_indices = {value: np.flatnonzero(regular & (year == value)) for value in (2023, 2024)}
    halves = {
        (value, half): index[:len(index)//2] if half == 1 else index[len(index)//2:]
        for value, index in regular_indices.items() for half in (1, 2)
    }
    transfers = (
        ("23h1_to_23h2", halves[(2023, 1)], halves[(2023, 2)]),
        ("23_to_24h1", regular_indices[2023], halves[(2024, 1)]),
        ("23_to_24h2", regular_indices[2023], halves[(2024, 2)]),
        ("24h1_to_24h2", halves[(2024, 1)], halves[(2024, 2)]),
    )

    screened = json.loads(
        (ROOT / "research/v24_exhaustive_transfer.json").read_text(encoding="utf-8")
    )["top"]
    # Keep the strongest configuration per feature.  This prevents a greedy
    # search from selecting many almost identical bin/shrink variants.
    candidates = {}
    for value in screened:
        if value["min_transfer"] <= 1.:
            continue
        candidates.setdefault((value["kind"], value["column"]), value)
    directions = {
        key: {
            label: direction(spec, source, valid, numeric, categorical, target, base)
            for label, source, valid in transfers
        }
        for key, spec in candidates.items()
    }
    total = {label: np.zeros(len(valid), dtype=float) for label, _source, valid in transfers}
    current_gains = {label: 0. for label, _source, _valid in transfers}
    selected = []
    remaining = dict(candidates)
    for step in range(MAX_STEPS):
        winner = None
        for key, spec in remaining.items():
            candidate_gains = {
                label: correction_gain(
                    target[valid], base[valid], total[label] + directions[key][label],
                )
                for label, _source, valid in transfers
            }
            # Require every transfer to stay positive and rank conservatively.
            if min(candidate_gains.values()) <= 0.:
                continue
            rank = (min(candidate_gains.values()), float(np.mean(list(candidate_gains.values()))))
            if winner is None or rank > winner[0]:
                winner = (rank, key, spec, candidate_gains)
        if winner is None:
            break
        old_rank = (min(current_gains.values()), float(np.mean(list(current_gains.values()))))
        if winner[0][0] <= old_rank[0] + .20:
            break
        _rank, key, spec, candidate_gains = winner
        for label in total:
            total[label] += directions[key][label]
        current_gains = candidate_gains
        selected.append({"step": step + 1, "spec": spec, "gains": candidate_gains})
        # Also remove close algebraic variants of the same feature family.
        family = spec["column"].replace("_s25", "").replace("_s100", "").replace("_rate", "")
        remaining = {
            candidate_key: candidate_spec
            for candidate_key, candidate_spec in remaining.items()
            if candidate_key != key and family not in candidate_spec["column"].replace("_s25", "").replace("_s100", "").replace("_rate", "")
        }

    # Full 2023 -> 2024 replay with F left exactly at v24.
    source = regular_indices[2023]
    valid = regular_indices[2024]
    full_correction = np.zeros(len(valid), dtype=float)
    for item in selected:
        full_correction += direction(
            item["spec"], source, valid, numeric, categorical, target, base,
        )
    regular_candidate = np.clip(base[valid] + full_correction, .005, .995)
    regular_detail = detail(
        target[valid], base[valid], regular_candidate,
        rows.iloc[valid].reset_index(drop=True),
    )
    all_2024 = np.flatnonzero(year == 2024)
    all_candidate = base[all_2024].copy()
    local_regular = rows.iloc[all_2024]["game_type"].eq("R").to_numpy()
    all_candidate[local_regular] = regular_candidate
    overall_detail = detail(
        target[all_2024], base[all_2024], all_candidate,
        rows.iloc[all_2024].reset_index(drop=True),
    )
    result = {
        "candidate_count": len(candidates), "selected": selected,
        "transfer_gains": current_gains,
        "regular_2024_detail": regular_detail,
        "overall_2024_detail": overall_detail,
    }
    output = ROOT / "research/v25_transfer_portfolio.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
