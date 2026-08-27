"""Forward-screen non-parametric calibration tables over the v24 probability."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v25_f_pair_transfer import context_frame, pair_codes
from research_v24_exhaustive_transfer import (
    encode_numeric, gain, table_direction,
)


ROOT = Path(__file__).resolve().parent
CONTEXTS = (
    "global", "count_state", "pressure_state", "balls_before", "strikes_before",
    "batter_hand", "pitcher_hand", "hand_matchup_code", "num_runners_on",
    "base_out_code", "inning_bucket", "score_bucket", "leverage_bucket",
)
BINS = (4, 8, 16, 32)
SHRINKS = (100., 400., 1600., 6400., 25600.)
SCALES = (.10, .25, .50, 1.)


def codes(base, context, source, valid, context_name, bins):
    if context_name == "global":
        return encode_numeric(base.iloc[source], base.iloc[valid], bins)
    return pair_codes(
        base.iloc[source], base.iloc[valid],
        context[context_name].iloc[source], context[context_name].iloc[valid], bins,
    )


def segment_gains(y, base, correction, rows):
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
        if active.sum() >= 40:
            masks[f"month_{int(month)}"] = active
    return {
        name: gain(y[active], base[active], correction[active])
        for name, active in masks.items()
    }


def main():
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    context_all = context_frame(raw)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    seasons = raw["season"].to_numpy(np.int16)
    positions = np.concatenate([
        np.flatnonzero(seasons == year) for year in (2023, 2024)
    ])
    rows = raw.iloc[positions].reset_index(drop=True)
    context = context_all.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    base_array = oof["blended"].astype(float)
    base = pd.Series(base_array)
    year = oof["season"].astype(int)
    if not np.allclose(y, rows["control_success"]):
        raise ValueError("v24 OOF rows do not align")

    output = {}
    for regime in ("R", "F"):
        active = rows["game_type"].eq(regime).to_numpy()
        indices = {value: np.flatnonzero(active & (year == value)) for value in (2023, 2024)}
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
        reports = []
        for context_name in CONTEXTS:
            for bins in BINS:
                for shrink in SHRINKS:
                    directions = {}
                    for label, source, valid in transfers:
                        encoded = codes(base, context, source, valid, context_name, bins)
                        directions[label] = table_direction(
                            encoded[0], encoded[1], y[source] - base_array[source], shrink,
                        )
                    for scale in SCALES:
                        values = {
                            label: gain(
                                y[valid], base_array[valid], scale * directions[label],
                            )
                            for label, _source, valid in transfers
                        }
                        if min(values.values()) > 0:
                            reports.append({
                                "context": context_name, "bins": bins,
                                "shrink": shrink, "scale": scale, "gains": values,
                                "min_transfer": min(values.values()),
                                "mean_transfer": float(np.mean(list(values.values()))),
                            })
        reports.sort(
            key=lambda item: (item["min_transfer"], item["mean_transfer"]), reverse=True,
        )
        strongest = {}
        for report in reports:
            strongest.setdefault(report["context"], report)
        source, valid = indices[2023], indices[2024]
        audits = []
        for report in strongest.values():
            encoded = codes(
                base, context, source, valid, report["context"], int(report["bins"]),
            )
            correction = table_direction(
                encoded[0], encoded[1], y[source] - base_array[source],
                float(report["shrink"]),
            ) * float(report["scale"])
            detail = segment_gains(
                y[valid], base_array[valid], correction,
                rows.iloc[valid].reset_index(drop=True),
            )
            audits.append({
                **report, "detail_2024": detail,
                "min_2024_segment": min(detail.values()),
            })
        audits.sort(
            key=lambda item: (
                item["min_2024_segment"], item["min_transfer"],
                item["detail_2024"]["all"],
            ), reverse=True,
        )
        output[regime] = {
            "positive_count": len(reports), "top": reports[:300], "audits": audits,
        }
    path = ROOT / "research/v25_probability_tables.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        regime: {
            "positive_count": item["positive_count"], "top": item["top"][:30],
            "audits": item["audits"],
        } for regime, item in output.items()
    }, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
