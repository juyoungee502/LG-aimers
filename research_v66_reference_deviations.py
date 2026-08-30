"""Audit four train-label context deviations reported by a public solution.

The reference's useful idea is a nested empirical-Bayes contrast: estimate a
pitcher's relative command against a batter side, then add smaller count and
runner refinements.  This implementation rebuilds every table from official
rows before the validation season and evaluates it over the stronger v64 OOF
anchor.  No reference artifact, prediction, or table value is consumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v66_hierarchical_residual import pitcher_bootstrap, report_segments
from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
TARGET = "control_success"
CLIP = (0.005, 0.995)
REFERENCE_WEIGHTS = np.asarray((0.20, 0.825, 0.280, 0.45), dtype=float)
OVERALL_SCALES = (0.25, 0.50, 0.75, 1.00)


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["pitcher_hand_key"] = (
        output["pitcher_id"].astype(str) + "|"
        + output["batter_hand"].astype(str)
    )
    output["advantage"] = (
        output["strikes_before"] > output["balls_before"]
    ).astype(np.int8)
    output["pitcher_hand_advantage_key"] = (
        output["pitcher_hand_key"] + "|" + output["advantage"].astype(str)
    )
    output["exact_count_key"] = (
        output["pitcher_hand_advantage_key"] + "|"
        + output["balls_before"].astype(str) + "|"
        + output["strikes_before"].astype(str)
    )
    output["runner_gate"] = output["num_runners_on"].gt(0).astype(np.int8)
    output["runner_key"] = (
        output["pitcher_hand_key"] + "|" + output["runner_gate"].astype(str)
    )
    return output


def nested_deviation(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    parent: str,
    child: str,
    shrinkage: float,
) -> np.ndarray:
    table = nested_deviation_table(history, parent, child, shrinkage)
    return rows[child].map(table).fillna(0.0).to_numpy(float)


def nested_deviation_table(
    history: pd.DataFrame,
    parent: str,
    child: str,
    shrinkage: float,
) -> pd.Series:
    """Return a frozen child-minus-parent command table."""
    parent_mean = history.groupby(parent, sort=False, observed=True)[TARGET].mean()
    grouped = history.groupby(child, sort=False, observed=True).agg(
        child_sum=(TARGET, "sum"), child_n=(TARGET, "size"),
        parent_key=(parent, "first"),
    )
    parent_rate = grouped["parent_key"].map(parent_mean)
    child_rate = grouped["child_sum"] / grouped["child_n"]
    grouped["deviation"] = (
        (child_rate - parent_rate)
        * grouped["child_n"] / (grouped["child_n"] + shrinkage)
    )
    return grouped["deviation"]


def fold_corrections(
    all_rows: pd.DataFrame, validation_year: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    history = all_rows.loc[all_rows["season"].lt(validation_year)]
    rows = all_rows.loc[all_rows["season"].eq(validation_year)].reset_index(drop=True)
    specifications = (
        ("pitcher_id", "pitcher_hand_key", 300.0, "platoon"),
        ("pitcher_hand_key", "pitcher_hand_advantage_key", 2000.0, "advantage"),
        ("pitcher_hand_advantage_key", "exact_count_key", 800.0, "count"),
        ("pitcher_hand_key", "runner_key", 2000.0, "runner"),
    )
    components = []
    audit: dict[str, object] = {}
    for parent, child, shrinkage, name in specifications:
        values = nested_deviation(history, rows, parent, child, shrinkage)
        components.append(values)
        audit[name] = {
            "shrinkage": shrinkage,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "coverage": float(np.mean(values != 0.0)),
        }
    matrix = np.column_stack(components)
    return rows[TARGET].to_numpy(float), matrix, audit


def gain(target: np.ndarray, anchor: np.ndarray, correction: np.ndarray) -> float:
    return float(
        bss(target, np.clip(anchor + correction, *CLIP)) - bss(target, anchor)
    )


def main() -> None:
    columns = [
        "season", "game_type", "pitcher_id", "batter_hand",
        "balls_before", "strikes_before", "num_runners_on", TARGET,
    ]
    raw = add_context(pd.read_csv(
        ROOT / "data/train.csv", usecols=columns,
        encoding="utf-8-sig", low_memory=False,
    ))
    with np.load(ROOT / "outputs/v64_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    target = oof["target"].astype(float)
    season = oof["season"].astype(int)
    anchor = oof["blended"].astype(float)
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(anchor) or not np.array_equal(
        rows[TARGET].to_numpy(float), target,
    ):
        raise ValueError("v64 OOF predictions do not align with train.csv")

    matrices: dict[int, np.ndarray] = {}
    audits: dict[str, object] = {}
    for year in (2023, 2024):
        fold_target, matrices[year], audits[str(year)] = fold_corrections(raw, year)
        if not np.array_equal(fold_target, target[season == year]):
            raise ValueError(f"target alignment failed in {year}")
    matrix = np.concatenate([matrices[year] for year in (2023, 2024)])
    regular = rows["game_type"].astype(str).eq("R").to_numpy()

    component_reports: dict[str, object] = {}
    names = ("platoon", "advantage", "count", "runner")
    for index, name in enumerate(names):
        correction = REFERENCE_WEIGHTS[index] * matrix[:, index]
        component_reports[name] = {
            str(year): {
                "gain": gain(
                    target[season == year], anchor[season == year],
                    correction[season == year],
                ),
                "regular_gain": gain(
                    target[(season == year) & regular],
                    anchor[(season == year) & regular],
                    correction[(season == year) & regular],
                ),
                "futures_gain": gain(
                    target[(season == year) & ~regular],
                    anchor[(season == year) & ~regular],
                    correction[(season == year) & ~regular],
                ),
            } for year in (2023, 2024)
        }

    reference_direction = matrix @ REFERENCE_WEIGHTS
    scale_reports: dict[str, object] = {}
    candidates: dict[float, np.ndarray] = {}
    for scale in OVERALL_SCALES:
        candidate = np.clip(anchor + scale * reference_direction, *CLIP)
        candidates[scale] = candidate
        scale_reports[str(scale)] = {
            str(year): report_segments(
                target[season == year], anchor[season == year],
                candidate[season == year], regular[season == year],
            ) for year in (2023, 2024)
        }
    selected = candidates[0.50]
    selected_reports = scale_reports["0.5"]
    bootstrap = {
        str(year): pitcher_bootstrap(
            target[season == year], anchor[season == year],
            selected[season == year],
            rows.loc[season == year, "pitcher_id"].to_numpy(),
            20000, 672000 + year,
        ) for year in (2023, 2024)
    }
    strict_gate = bool(
        all(selected_reports[str(year)]["gain"] > 0 for year in (2023, 2024))
        and all(bootstrap[str(year)]["positive_probability"] >= 0.80
                for year in (2023, 2024))
    )
    report = {
        "baseline": "v64_public_1135_1_anchor",
        "candidate": "public_reference_nested_label_deviations_clean_room",
        "reference_weights": REFERENCE_WEIGHTS.tolist(),
        "selected_overall_scale": 0.50,
        "fold_feature_audit": audits,
        "component_reports": component_reports,
        "scale_reports": scale_reports,
        "selected_reports": selected_reports,
        "bootstrap": bootstrap,
        "strict_gate": strict_gate,
        "rules": {
            "strict_prior_seasons_only": True,
            "official_train_only": True,
            "external_table_model_or_prediction_used": False,
            "forbidden_2025_trackman_used": False,
            "test_row_aggregation_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    (ROOT / "research/v66_reference_deviations.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/v66_reference_deviations_oof.npz",
        target=target, season=season, anchor=anchor,
        components=matrix, reference_direction=reference_direction,
        blended=selected,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
