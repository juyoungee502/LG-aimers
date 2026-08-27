"""Audit uncertainty-normalized entity residual tables over the v23 base.

The fitted value is a regularized one-step Newton offset in logit space.  A
table is always learned from labelled OOF rows in an earlier time block and
then frozen before it is applied to a later block.  No validation or test-row
aggregate is used to construct a feature.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SHRINKS = (25., 50., 100., 200., 400., 800., 1600., 3200., 6400., 12800.)
# Coarse screen first.  A later script can refine the neighbourhood of any
# direction that survives the full time-block audit.
WEIGHTS = np.round(np.arange(-.25, 1.2501, .25), 4)
GROUPS = {
    "r_batter_pitcher_hand": ("batter_id", "pitcher_hand"),
    "r_pitcher_batter_hand": ("pitcher_id", "batter_hand"),
    "r_pitcher_batter_hand_count": (
        "pitcher_id", "batter_hand", "count_state",
    ),
    "r_pitcher_batter": ("pitcher_id", "batter_id"),
    "r_pitcher": ("pitcher_id",),
    "r_batter": ("batter_id",),
    "r_pitcher_team_batter_hand": ("pitcher_team_id", "batter_hand"),
    "r_batter_team_pitcher_hand": ("batter_team_id", "pitcher_hand"),
}


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1. - 1e-6)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    value = np.asarray(value, dtype=float)
    return 1. / (1. + np.exp(-value))


def bss_gain(target, base, candidate):
    rate = float(np.mean(target))
    reference = rate * (1. - rate)
    old = np.mean((np.asarray(target) - np.asarray(base)) ** 2)
    new = np.mean((np.asarray(target) - np.asarray(candidate)) ** 2)
    return float(100000. * (old - new) / reference)


def fit_sufficient_statistics(frame, target, base, keys):
    regular = frame["game_type"].eq("R").to_numpy()
    work = frame.loc[regular, list(keys)].copy()
    probability = np.clip(np.asarray(base)[regular], 1e-6, 1. - 1e-6)
    residual = np.asarray(target)[regular] - probability
    # Do not let a season-wide calibration shift masquerade as an entity effect.
    residual -= float(np.mean(residual))
    work["residual"] = residual
    work["information"] = probability * (1. - probability)
    table = work.groupby(list(keys), observed=True, sort=False).agg(
        residual_sum=("residual", "sum"),
        information_sum=("information", "sum"),
        n=("information", "size"),
    ).reset_index()
    return table, float(work["information"].mean())


def apply_offset(frame, table, keys, shrink, mean_information):
    query = frame[list(keys)].copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(table, on=list(keys), how="left", sort=False)
    query = query.sort_values("_order")
    numerator = query["residual_sum"].fillna(0.).to_numpy(float)
    denominator = query["information_sum"].fillna(0.).to_numpy(float)
    offset = numerator / (denominator + shrink * mean_information)
    offset = np.clip(offset, -2., 2.)
    offset[~frame["game_type"].eq("R").to_numpy()] = 0.
    return offset


def sections(length):
    edges = np.linspace(0, length, 5, dtype=int)
    return edges


def main():
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
        usecols=[
            "season", "game_type", "pitcher_id", "batter_id",
            "pitcher_hand", "batter_hand", "pitcher_team_id",
            "batter_team_id", "balls_before", "strikes_before",
            "control_success",
        ],
    )
    raw["count_state"] = (
        pd.to_numeric(raw["balls_before"], errors="coerce").fillna(-1).astype(int) * 3
        + pd.to_numeric(raw["strikes_before"], errors="coerce").fillna(-1).astype(int)
    )
    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    with np.load(
        ROOT / "research/rolling_2022_2024/v11_oof_predictions.npz"
    ) as archive:
        v11 = {key: archive[key] for key in archive.files}

    folds = {}
    for year in (2022, 2023, 2024):
        source = v11 if year == 2022 else oof
        mask = source["season"] == year
        frame = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        target = source["target"][mask].astype(float)
        if len(frame) != len(target):
            raise ValueError(f"v23/raw row mismatch for {year}: {len(frame)} != {len(target)}")
        if not np.array_equal(frame["control_success"].to_numpy(), target):
            raise ValueError(f"v23/raw target mismatch for {year}")
        folds[year] = {
            "frame": frame,
            "target": target,
            "base": np.clip(source["blended"][mask].astype(float), .005, .995),
            "edges": sections(len(frame)),
            "base_version": "v11" if year == 2022 else "v23",
        }

    # Every evaluation block is strictly later than the rows used to fit its table.
    source_specs = {
        "2022_full": (2022, slice(None)),
        "2023_full": (2023, slice(None)),
        "2023_h1": (2023, slice(0, folds[2023]["edges"][2])),
        "2024_h1": (2024, slice(0, folds[2024]["edges"][2])),
    }
    block_specs = {
        "2022_to_2023": (2023, slice(None), "2022_full"),
        "2023_h1_to_h2": (
            2023, slice(folds[2023]["edges"][2], None), "2023_h1",
        ),
        "2023_to_2024": (2024, slice(None), "2023_full"),
        "2024_h1_to_h2": (
            2024, slice(folds[2024]["edges"][2], None), "2024_h1",
        ),
    }
    for quarter in range(4):
        block_specs[f"2023_to_2024_q{quarter + 1}"] = (
            2024,
            slice(folds[2024]["edges"][quarter], folds[2024]["edges"][quarter + 1]),
            "2023_full",
        )

    candidates = []
    for group_name, keys in GROUPS.items():
        tables = {}
        for source_name, (year, section) in source_specs.items():
            fold = folds[year]
            table, mean_information = fit_sufficient_statistics(
                fold["frame"].iloc[section].reset_index(drop=True),
                fold["target"][section], fold["base"][section], keys,
            )
            tables[source_name] = (table, mean_information)

        for shrink in SHRINKS:
            offsets = {}
            for label, (year, section, source_name) in block_specs.items():
                table, mean_information = tables[source_name]
                offsets[label] = apply_offset(
                    folds[year]["frame"].iloc[section].reset_index(drop=True),
                    table, keys, shrink, mean_information,
                )
            for weight in WEIGHTS:
                gains = {}
                for label, (year, section, _source_name) in block_specs.items():
                    fold = folds[year]
                    base = fold["base"][section]
                    candidate = sigmoid(logit(base) + weight * offsets[label])
                    gains[label] = bss_gain(fold["target"][section], base, candidate)
                primary = [gains["2022_to_2023"], gains["2023_to_2024"]]
                quarters = [gains[f"2023_to_2024_q{i}"] for i in range(1, 5)]
                candidates.append({
                    "group": group_name,
                    "keys": list(keys),
                    "shrink": shrink,
                    "weight": float(weight),
                    "gains": gains,
                    "min_year_forward": float(min(primary)),
                    "mean_year_forward": float(np.mean(primary)),
                    "min_half_forward": float(min(
                        gains["2023_h1_to_h2"], gains["2024_h1_to_h2"],
                    )),
                    "min_2024_quarter": float(min(quarters)),
                })

    by_year = sorted(
        candidates,
        key=lambda row: (row["min_year_forward"], row["mean_year_forward"]),
        reverse=True,
    )
    strict = sorted(
        candidates,
        key=lambda row: (
            min(row["min_year_forward"], row["min_half_forward"],
                row["min_2024_quarter"]),
            row["mean_year_forward"],
        ),
        reverse=True,
    )
    report = {
        "method": "regularized one-step Newton entity offsets in logit space",
        "selection_note": (
            "Primary ranking uses full-season forward transfer; strict ranking also "
            "requires both half-season transfers and all 2024 quarters. The 2022 "
            "source uses archived v11 OOF because v23 OOF starts in 2023."
        ),
        "best_year_forward": by_year[:100],
        "best_strict": strict[:100],
    }
    output = ROOT / "research/v33_newton_entity_residual.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_year_forward": by_year[:10],
        "best_strict": strict[:10],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
