"""Screen frozen entity-context residual tables on top of v16."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import TARGET, bss


SPECS = {
    "pitcher_count": ["pitcher_id", "count_state"],
    "pitcher_bhand": ["pitcher_id", "batter_hand"],
    "pitcher_game": ["pitcher_id", "game_type"],
    "batter_count": ["batter_id", "count_state"],
    "batter_phand": ["batter_id", "pitcher_hand"],
    "pitcher_team_count": ["pitcher_team_id", "count_state"],
    "batter_team_count": ["batter_team_id", "count_state"],
}


def build_table(frame, residual, keys, shrink):
    work = frame[keys].copy()
    work["residual"] = residual
    table = work.groupby(keys, observed=True, sort=False)["residual"].agg(
        ["sum", "size"]
    ).reset_index()
    table["value"] = table["sum"] / (table["size"] + shrink)
    mapped = apply_table(frame, table, keys)
    table["value"] -= float(mapped.mean())
    return table[keys + ["value"]]


def apply_table(frame, table, keys):
    left = frame[keys].copy()
    left["_order"] = np.arange(len(left))
    got = left.merge(table, on=keys, how="left", sort=False).sort_values("_order")
    return got["value"].fillna(0.).to_numpy(float)


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    rows = pd.concat([
        data.loc[data["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    rows["count_state"] = (
        rows["balls_before"] * 3 + rows["strikes_before"]
    ).astype(np.int8)
    with np.load(root / "outputs" / "v16_oof_predictions.npz", allow_pickle=False) as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    if len(rows) != len(oof["target"]) or not np.allclose(
        rows[TARGET].to_numpy(), oof["target"]
    ):
        raise ValueError("v16 OOF and train.csv do not align")
    source = oof["season"] == 2023
    valid = oof["season"] == 2024
    source_rows = rows.loc[source].reset_index(drop=True)
    valid_rows = rows.loc[valid].reset_index(drop=True)
    source_residual = oof["target"][source] - oof["blended"][source]
    y = oof["target"][valid].astype(float)
    base = oof["blended"][valid].astype(float)
    halfway = len(y) // 2

    reports = []
    for name, keys in SPECS.items():
        for shrink in (100., 200., 400., 800., 1600., 3200., 6400., 12800.):
            table = build_table(source_rows, source_residual, keys, shrink)
            raw = apply_table(valid_rows, table, keys)
            for regular_only in (False, True):
                values = raw.copy()
                if regular_only:
                    values[valid_rows["game_type"].ne("R").to_numpy()] = 0.
                for weight in np.arange(-.5, 2.001, .05):
                    prediction = np.clip(base + weight * values, .005, .995)
                    gains = [
                        bss(y, prediction) - bss(y, base),
                        bss(y[:halfway], prediction[:halfway])
                        - bss(y[:halfway], base[:halfway]),
                        bss(y[halfway:], prediction[halfway:])
                        - bss(y[halfway:], base[halfway:]),
                    ]
                    reports.append({
                        "name": name, "keys": keys, "shrink": shrink,
                        "weight": float(weight), "regular_only": regular_only,
                        "gain_2024": gains[0], "gain_first_half": gains[1],
                        "gain_second_half": gains[2],
                        "min_half": min(gains[1:]),
                    })
    reports.sort(
        key=lambda item: (item["min_half"], item["gain_2024"]), reverse=True
    )
    output = root / "research" / "residual_interactions.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"top": reports}, indent=2), encoding="utf-8")
    print(json.dumps(reports[:40], indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
