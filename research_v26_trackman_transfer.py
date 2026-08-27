"""Strict forward screen of allowed 2019-2024 Trackman residual tables.

This is a research-only screen.  Trackman features are built from seasons
strictly before each row, and all candidate directions are measured over the
same four chronological transfers used by the v25 portfolio.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_v25_f_pair_transfer import context_frame, pair_codes, segment_detail
from research_v24_exhaustive_transfer import (
    encode_numeric, gain, table_direction,
)
from trackman_context import (
    FEATURE_COLUMNS, attach_context, pitcher_mapping, prepare_trackman,
)


ROOT = Path(__file__).resolve().parent
BIN_COUNTS = (4, 8)
SHRINKS = (100., 400., 1600., 6400.)
SCALES = (.25, .5, 1.)
CONTEXT_COLUMNS = (
    "count_state", "pressure_state", "balls_before", "strikes_before",
    "batter_hand", "pitcher_hand", "hand_matchup_code", "num_runners_on",
    "base_out_code", "inning_bucket", "score_bucket", "leverage_bucket",
)


def directions_for(spec, numeric, context, source, valid, residual):
    if spec["type"] == "one_d":
        codes = encode_numeric(
            numeric[spec["column"]].iloc[source],
            numeric[spec["column"]].iloc[valid], int(spec["bins"]),
        )
    else:
        codes = pair_codes(
            numeric[spec["column"]].iloc[source],
            numeric[spec["column"]].iloc[valid],
            context[spec["context"]].iloc[source],
            context[spec["context"]].iloc[valid], int(spec["bins"]),
        )
    if codes is None:
        return None
    return table_direction(
        codes[0], codes[1], residual[source], float(spec["shrink"]),
    )


def screen_regime(rows, numeric, context, y, base, year, regime):
    active = rows["game_type"].eq(regime).to_numpy()
    indices = {
        value: np.flatnonzero(active & (year == value)) for value in (2023, 2024)
    }
    halves = {
        (value, half): index[:len(index)//2] if half == 1 else index[len(index)//2:]
        for value, index in indices.items() for half in (1, 2)
    }
    transfers = (
        ("23h1_to_23h2", halves[(2023, 1)], halves[(2023, 2)]),
        ("23_to_24h1", indices[2023], halves[(2024, 1)]),
        ("23_to_24h2", indices[2023], halves[(2024, 2)]),
        ("24h1_to_24h2", halves[(2024, 1)], halves[(2024, 2)]),
    )
    residual = y - base
    geometry = [
        {"type": kind, "column": column, "context": context_name,
         "bins": bins, "shrink": shrink}
        for column in FEATURE_COLUMNS
        for kind, context_name in (
            [("one_d", None)]
            + [("pair", value) for value in CONTEXT_COLUMNS]
        )
        for bins in BIN_COUNTS for shrink in SHRINKS
    ]
    reports = []
    for index, spec in enumerate(geometry):
        raw = {}
        for label, source, valid in transfers:
            raw[label] = directions_for(
                spec, numeric, context, source, valid, residual,
            )
            if raw[label] is None:
                break
        if len(raw) != len(transfers):
            continue
        for scale in SCALES:
            gains = {
                label: gain(y[valid], base[valid], scale * raw[label])
                for label, _source, valid in transfers
            }
            if min(gains.values()) > 0.:
                reports.append({
                    **spec, "scale": scale, "gains": gains,
                    "min_transfer": min(gains.values()),
                    "mean_transfer": float(np.mean(list(gains.values()))),
                })
        if (index + 1) % 200 == 0:
            print(
                f"{regime}: screened {index + 1}/{len(geometry)}",
                flush=True,
            )
    reports.sort(
        key=lambda item: (item["min_transfer"], item["mean_transfer"]),
        reverse=True,
    )

    strongest = {}
    for report in reports:
        strongest.setdefault(
            (report["type"], report["column"], report.get("context")), report,
        )
    source, valid = indices[2023], indices[2024]
    audited = []
    for report in strongest.values():
        raw = directions_for(report, numeric, context, source, valid, residual)
        correction = float(report["scale"]) * raw
        details = segment_detail(
            y[valid], base[valid], correction,
            rows.iloc[valid].reset_index(drop=True),
        )
        audited.append({
            **report, "gain_2024": details["all"],
            "min_segment_2024": min(details.values()),
            "segments_2024": details,
        })
    audited.sort(
        key=lambda item: (
            item["min_segment_2024"], item["min_transfer"], item["gain_2024"],
        ), reverse=True,
    )
    return {
        "screened": len(geometry), "positive_all_four": len(reports),
        "unique_positive": len(strongest), "top_transfer": reports[:250],
        "top_audited": audited[:250],
    }


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    trackman = pd.read_csv(
        ROOT / "data/trackman_history.csv",
        usecols=[
            "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
            "balls_before", "strikes_before", "batter_hand", "rel_speed",
        ], encoding="utf-8-sig", low_memory=False,
    )
    if int(trackman["season"].max()) > 2024:
        raise ValueError("Forbidden post-2024 Trackman rows detected")
    mapping, mapping_report = pitcher_mapping(ROOT, raw, trackman)
    trackman_features = attach_context(raw, prepare_trackman(trackman, mapping))

    bases = training_history_arrays(raw, target_series)
    numeric_all = engineer_features(
        raw, *bases, global_prior=float(target_series.mean()),
    )
    add_training_component_features(numeric_all, raw)
    numeric_all = add_state_interactions(numeric_all)
    numeric_all = pd.concat([numeric_all, trackman_features], axis=1)
    contexts_all = context_frame(raw)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    seasons = raw["season"].to_numpy(np.int16)
    positions = np.concatenate([
        np.flatnonzero(seasons == value) for value in (2023, 2024)
    ])
    if not np.allclose(target_all[positions], oof["target"]):
        raise ValueError("v24 OOF rows do not align")
    rows = raw.iloc[positions].reset_index(drop=True)
    numeric = numeric_all.iloc[positions].reset_index(drop=True)
    context = contexts_all.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    year = oof["season"].astype(int)
    result = {
        "feature_columns": list(FEATURE_COLUMNS),
        "mapped_pitchers": len(mapping),
        "minimum_mapping_confidence": float(mapping_report["confidence"].min()),
        "forbidden_2025_trackman_used": False,
        "coverage_2024": {
            column: float(numeric.loc[year == 2024, column].notna().mean())
            for column in FEATURE_COLUMNS
        },
    }
    for regime in ("R", "F"):
        result[regime] = screen_regime(
            rows, numeric, context, y, base, year, regime,
        )
        print(json.dumps({
            "regime": regime,
            "positive_all_four": result[regime]["positive_all_four"],
            "top_audited": result[regime]["top_audited"][:10],
        }, indent=2), flush=True)
    path = ROOT / "research/v26_trackman_transfer.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
