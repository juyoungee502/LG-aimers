"""Add a frozen prior-season pitch-choice failure correction to v15."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import (
    DETAILS, TARGET, bss, reconstruct_labels, signal,
)


OUTCOME_K = 500.0
SELECTION_K = 300.0
FAILURE_WEIGHTS = np.array([-.75, 1.50, -.50], dtype=np.float64)
OUTER_WEIGHT = 1.85


def history_frame(data, labels):
    history = data[[
        "season", "pitcher_id", "batter_hand", "balls_before", "strikes_before",
        "game_type", TARGET,
    ]].copy()
    history["count_state"] = (
        history["balls_before"] * 3 + history["strikes_before"]
    ).astype(np.int8)
    history["pitch_type"] = labels["pitch_type"]
    for name in DETAILS:
        history[name] = labels[name]
    return history


def raw_failure_signal(source, rows):
    components = [
        signal(
            source, rows, name, OUTCOME_K, SELECTION_K, "hand_count"
        )[1]
        for name in DETAILS
    ]
    return np.column_stack(components) @ FAILURE_WEIGHTS


def correction_for_year(history, year):
    source = history.loc[history["season"].lt(year)]
    rows = history.loc[history["season"].eq(year)].reset_index(drop=True)
    proxy = history.loc[history["season"].eq(year - 1)].reset_index(drop=True)
    raw = raw_failure_signal(source, rows)
    proxy_raw = raw_failure_signal(source, proxy)
    proxy_regular = proxy["game_type"].eq("R").to_numpy()
    center = float(proxy_raw[proxy_regular].mean())
    regular = rows["game_type"].eq("R").to_numpy()
    correction = np.zeros(len(rows), dtype=np.float64)
    correction[regular] = OUTER_WEIGHT * (raw[regular] - center)
    return correction, center


def production_table(history):
    pitchers = np.sort(history["pitcher_id"].unique())
    grid = pd.MultiIndex.from_product(
        [pitchers, (1, 2), range(12)],
        names=["pitcher_id", "batter_hand", "count_state"],
    ).to_frame(index=False)
    grid["balls_before"] = (grid["count_state"] // 3).astype(np.int8)
    grid["strikes_before"] = (grid["count_state"] % 3).astype(np.int8)
    raw_grid = raw_failure_signal(history, grid)

    proxy = history.loc[history["season"].eq(int(history["season"].max()))].reset_index(drop=True)
    raw_proxy = raw_failure_signal(history, proxy)
    proxy_regular = proxy["game_type"].eq("R").to_numpy()
    center = float(raw_proxy[proxy_regular].mean())
    delta = OUTER_WEIGHT * (raw_grid - center)
    return {
        "outcome_k": OUTCOME_K,
        "selection_k": SELECTION_K,
        "failure_weights": dict(zip(DETAILS, FAILURE_WEIGHTS.tolist())),
        "outer_weight": OUTER_WEIGHT,
        "center": center,
        "keys": [
            f"{int(pitcher)}:{int(hand)}:{int(count)}"
            for pitcher, hand, count in grid[
                ["pitcher_id", "batter_hand", "count_state"]
            ].itertuples(index=False, name=None)
        ],
        "deltas": delta.astype(np.float32).tolist(),
        "game_type": "R",
    }


def main():
    root = Path(__file__).resolve().parent
    metadata_path = root / "submit" / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v15_weighted_categorical_specialist":
        raise ValueError(f"Expected v15 artifacts, found {metadata.get('version')}")
    with np.load(root / "outputs" / "v15_oof_predictions.npz", allow_pickle=False) as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    data = pd.read_csv(root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    labels = reconstruct_labels(data)
    history = history_frame(data, labels)
    oof_rows = pd.concat([
        data.loc[data["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(oof_rows) != len(oof["target"]) or not np.allclose(
        oof_rows[TARGET].to_numpy(), oof["target"]
    ):
        raise ValueError("v15 OOF and train.csv do not align")

    upgraded = oof["blended"].astype(np.float64).copy()
    reports = {}
    for year in (2023, 2024):
        mask = oof["season"] == year
        correction, center = correction_for_year(history, year)
        before = bss(oof["target"][mask], upgraded[mask])
        upgraded[mask] = np.clip(upgraded[mask] + correction, .005, .995)
        after = bss(oof["target"][mask], upgraded[mask])
        reports[str(year)] = {
            "before_bss": before, "after_bss": after,
            "gain": after - before, "fixed_proxy_center": center,
        }
        print(
            f"v16 fixed-center {year}: {before:.4f} -> {after:.4f} "
            f"({after-before:+.4f}); center={center:+.8f}", flush=True,
        )

    final_table = production_table(history)
    metadata["version"] = "v16_pitch_failure_prior"
    metadata["pitch_failure_prior"] = final_table
    metadata["training_info"]["v16_validation"] = {
        **reports,
        "pitch_type_reconstruction_coverage": float(labels["pitch_type"].notna().mean()),
        "detail_reconstruction_coverage": float(
            labels[list(DETAILS)].notna().all(axis=1).mean()
        ),
        "row_independent_inference": True,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        root / "outputs" / "v16_oof_predictions.npz",
        **{key: value for key, value in oof.items() if key != "blended"},
        blended=upgraded,
    )
    print(
        f"Stored {len(final_table['keys'])} frozen pitch-failure cells; "
        f"production center={final_table['center']:+.8f}"
    )


if __name__ == "__main__":
    main()
