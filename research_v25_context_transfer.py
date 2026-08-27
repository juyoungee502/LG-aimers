"""Exhaustive time-forward screen of hierarchical context residuals over v24.

Every source table is learned from labelled training rows only.  Deployment can
freeze the same table from 2024 and map one evaluation row at a time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent
SHRINKS = (25., 50., 100., 200., 400., 800., 1600., 3200., 6400.)
SCALES = (.25, .50, .75, 1.00, 1.25)


CONTEXTS = {
    # Low-cardinality state effects.
    "count": (("count_state",), None),
    "hands": (("pitcher_hand", "batter_hand"), None),
    "count_hands": (("count_state", "pitcher_hand", "batter_hand"), ("count_state",)),
    "pressure_hands": (("pressure_state", "pitcher_hand", "batter_hand"), ("pressure_state",)),
    "base_out": (("base_state", "outs_before"), None),
    "runner_count": (("num_runners_on",), None),
    "inning_state": (("inning_bucket", "top_bottom"), None),
    "score_state": (("score_bucket", "is_pitcher_home"), None),
    "leverage_state": (("leverage_bucket", "pressure_state"), None),
    # Entity main effects and stable conditional deviations.
    "pitcher": (("pitcher_id",), None),
    "pitcher_hand": (("pitcher_id", "batter_hand"), ("pitcher_id",)),
    "pitcher_count": (("pitcher_id", "count_state"), ("pitcher_id",)),
    "pitcher_pressure": (("pitcher_id", "pressure_state"), ("pitcher_id",)),
    "pitcher_runners": (("pitcher_id", "num_runners_on"), ("pitcher_id",)),
    "pitcher_base_out": (("pitcher_id", "base_state", "outs_before"), ("pitcher_id",)),
    "pitcher_inning": (("pitcher_id", "inning_bucket"), ("pitcher_id",)),
    "pitcher_score": (("pitcher_id", "score_bucket"), ("pitcher_id",)),
    "pitcher_leverage": (("pitcher_id", "leverage_bucket"), ("pitcher_id",)),
    "pitcher_hand_count": (
        ("pitcher_id", "batter_hand", "count_state"),
        ("pitcher_id", "batter_hand"),
    ),
    "pitcher_hand_pressure": (
        ("pitcher_id", "batter_hand", "pressure_state"),
        ("pitcher_id", "batter_hand"),
    ),
    "pitcher_hand_runners": (
        ("pitcher_id", "batter_hand", "num_runners_on"),
        ("pitcher_id", "batter_hand"),
    ),
    "batter": (("batter_id",), None),
    "batter_hand": (("batter_id", "pitcher_hand"), ("batter_id",)),
    "batter_count": (("batter_id", "count_state"), ("batter_id",)),
    "batter_pressure": (("batter_id", "pressure_state"), ("batter_id",)),
    "batter_runners": (("batter_id", "num_runners_on"), ("batter_id",)),
    "batter_base_out": (("batter_id", "base_state", "outs_before"), ("batter_id",)),
    "batter_hand_count": (
        ("batter_id", "pitcher_hand", "count_state"),
        ("batter_id", "pitcher_hand"),
    ),
    "pitcher_team": (("pitcher_team_id",), None),
    "batter_team": (("batter_team_id",), None),
    "team_matchup": (("pitcher_team_id", "batter_team_id"), ("pitcher_team_id",)),
    "pitcher_team_count": (("pitcher_team_id", "count_state"), ("pitcher_team_id",)),
    "batter_team_count": (("batter_team_id", "count_state"), ("batter_team_id",)),
}


def prepare(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["count_state"] = (
        out["balls_before"].to_numpy(np.int16) * 3
        + out["strikes_before"].to_numpy(np.int16)
    )
    balls = out["balls_before"].to_numpy(np.int16)
    strikes = out["strikes_before"].to_numpy(np.int16)
    out["pressure_state"] = np.where(
        (balls == 3) & (strikes == 2), 2,
        np.where((balls == 3) | (strikes == 2), 1, 0),
    ).astype(np.int8)
    out["inning_bucket"] = np.minimum(out["inning"].fillna(0).to_numpy(np.int16), 10)
    score = out["score_diff_pitcher_team"].fillna(0).to_numpy(float)
    out["score_bucket"] = np.clip(score, -3, 3).astype(np.int8)
    leverage = out["li"].fillna(0).to_numpy(float)
    out["leverage_bucket"] = np.digitize(leverage, (0.5, 1.0, 2.0, 4.0)).astype(np.int8)
    top = out["top_bottom"].astype(str).eq("T").to_numpy()
    home_diff = out["score_diff_home"].fillna(0).to_numpy(float)
    out["is_pitcher_home"] = np.where(top, score == home_diff, score == -home_diff).astype(np.int8)
    return out


def mapped_stat(source, query, keys, values, statistic):
    frame = source[list(keys)].copy()
    frame["_value"] = values
    grouped = frame.groupby(list(keys), observed=True, sort=False)["_value"].agg(
        ["sum", "count", "mean"]
    )
    index = pd.MultiIndex.from_frame(query[list(keys)]) if len(keys) > 1 \
        else pd.Index(query[keys[0]], name=keys[0])
    return grouped[statistic].reindex(index).to_numpy(float)


def table_direction(source, query, residual, child_keys, parent_keys, shrink):
    child_sum = mapped_stat(source, query, child_keys, residual, "sum")
    child_n = mapped_stat(source, query, child_keys, residual, "count")
    child_sum = np.nan_to_num(child_sum)
    child_n = np.nan_to_num(child_n)
    if parent_keys is None:
        return child_sum / (child_n + shrink)
    child_mean = np.divide(
        child_sum, child_n, out=np.zeros(len(query), dtype=float), where=child_n > 0,
    )
    parent_mean = mapped_stat(source, query, parent_keys, residual, "mean")
    parent_mean = np.nan_to_num(parent_mean)
    return child_n / (child_n + shrink) * (child_mean - parent_mean)


def geometry(target, base, correction):
    uncertainty = float(target.mean() * (1. - target.mean()))
    residual = target - base
    return (
        200000. * float(np.mean(residual * correction)) / uncertainty,
        100000. * float(np.mean(correction * correction)) / uncertainty,
    )


def exact_gain(target, base, correction):
    candidate = np.clip(base + correction, .005, .995)
    return bss(target, candidate) - bss(target, base)


def segment_masks(rows):
    position = np.arange(len(rows))
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "half_1": position < len(rows) // 2,
        "half_2": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
    }
    for month in sorted(rows["game_month"].unique()):
        active = rows["game_month"].eq(month).to_numpy()
        if active.sum() >= 100:
            masks[f"month_{int(month)}"] = active
    return masks


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    data = prepare(raw)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    positions = np.concatenate([
        np.flatnonzero(data["season"].to_numpy() == year) for year in (2023, 2024)
    ])
    rows = data.iloc[positions].reset_index(drop=True)
    target = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    year = oof["season"].astype(int)
    if not np.allclose(target, raw["control_success"].to_numpy(float)[positions]):
        raise ValueError("v24 OOF rows do not align")
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

    reports = []
    for context_name, (child_keys, parent_keys) in CONTEXTS.items():
        for shrink in SHRINKS:
            curves = {}
            for label, source_index, valid_index in transfers:
                correction = table_direction(
                    rows.iloc[source_index], rows.iloc[valid_index],
                    target[source_index] - base[source_index], child_keys, parent_keys, shrink,
                )
                curves[label] = geometry(target[valid_index], base[valid_index], correction)
            for scale in SCALES:
                gains = {
                    label: linear * scale - quadratic * scale * scale
                    for label, (linear, quadratic) in curves.items()
                }
                if min(gains.values()) > 0:
                    reports.append({
                        "context": context_name, "child_keys": list(child_keys),
                        "parent_keys": list(parent_keys) if parent_keys else None,
                        "shrink": shrink, "scale": scale, "gains": gains,
                        "min_transfer": min(gains.values()),
                        "mean_transfer": float(np.mean(list(gains.values()))),
                    })
        print(f"screened {context_name}", flush=True)

    reports.sort(key=lambda item: (item["min_transfer"], item["mean_transfer"]), reverse=True)
    # Audit the strongest variant per context on the complete 2023 -> 2024
    # replay, so annual gains cannot hide a bad month.
    strongest = {}
    for report in reports:
        strongest.setdefault(report["context"], report)
    source_index, valid_index = regular_indices[2023], regular_indices[2024]
    audits = []
    masks = segment_masks(rows.iloc[valid_index].reset_index(drop=True))
    for report in strongest.values():
        direction = table_direction(
            rows.iloc[source_index], rows.iloc[valid_index],
            target[source_index] - base[source_index],
            tuple(report["child_keys"]),
            tuple(report["parent_keys"]) if report["parent_keys"] else None,
            float(report["shrink"]),
        ) * float(report["scale"])
        detail = {
            name: exact_gain(target[valid_index][active], base[valid_index][active], direction[active])
            for name, active in masks.items()
        }
        audits.append({**report, "detail_2024": detail, "min_2024_segment": min(detail.values())})
    audits.sort(
        key=lambda item: (item["min_2024_segment"], item["min_transfer"], item["detail_2024"]["all"]),
        reverse=True,
    )
    result = {
        "tested_contexts": len(CONTEXTS), "positive_configurations": len(reports),
        "top_transfer": reports[:300], "context_audits": audits,
    }
    output = ROOT / "research/v25_context_transfer.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "tested_contexts": len(CONTEXTS), "positive_configurations": len(reports),
        "top_transfer": reports[:20], "context_audits": audits[:20],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
