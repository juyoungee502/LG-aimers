"""Screen row-independent player structure corrections for v59.

The current public-feedback directions are essentially saturated.  This audit
looks for a genuinely new direction while keeping the competition constraints:
all tables are frozen from source labels and each validation/test row is only a
lookup.  Raw player levels, player-by-opponent-hand differentials and exposure
shapes are evaluated on forward transfers before anything is promoted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
CLIP = (0.005, 0.995)


@dataclass(frozen=True)
class Split:
    name: str
    source: np.ndarray
    validation: np.ndarray


def key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    result = frame[columns[0]].astype(str)
    for column in columns[1:]:
        result = result.str.cat(frame[column].astype(str), sep="|")
    return result


def frozen_category_direction(
    source: pd.DataFrame,
    validation: pd.DataFrame,
    residual: np.ndarray,
    columns: tuple[str, ...],
    shrinkage: float,
    parent: tuple[str, ...] | None = None,
) -> np.ndarray:
    work = source.loc[:, list(dict.fromkeys(columns + (parent or ())))].copy()
    centered = residual - float(np.mean(residual))
    if parent:
        parent_key = key(work, parent)
        parent_frame = pd.DataFrame({"key": parent_key, "value": centered})
        parent_mean = parent_frame.groupby("key", sort=False)["value"].transform("mean")
        centered = centered - parent_mean.to_numpy(float)
    source_key = key(work, columns)
    stats = pd.DataFrame({"key": source_key, "value": centered}).groupby(
        "key", sort=False,
    )["value"].agg(["sum", "count"])
    lookup = stats["sum"] / (stats["count"] + shrinkage)
    return key(validation, columns).map(lookup).fillna(0.0).to_numpy(float)


def quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def frozen_numeric_direction(
    source_values: np.ndarray,
    validation_values: np.ndarray,
    residual: np.ndarray,
    bins: int,
    shrinkage: float,
) -> np.ndarray:
    edges = quantile_edges(source_values, bins)
    source_bin = np.digitize(source_values, edges[1:-1], right=True)
    validation_bin = np.digitize(validation_values, edges[1:-1], right=True)
    centered = residual - float(np.mean(residual))
    stats = pd.DataFrame({"bin": source_bin, "value": centered}).groupby(
        "bin", sort=False,
    )["value"].agg(["sum", "count"])
    lookup = stats["sum"] / (stats["count"] + shrinkage)
    return pd.Series(validation_bin).map(lookup).fillna(0.0).to_numpy(float)


def add_season_exposure(all_rows: pd.DataFrame) -> pd.DataFrame:
    out = all_rows.copy()
    for role in ("pitcher", "batter"):
        id_column = f"{role}_id"
        count_column = f"asof_{role}_n"
        end = (
            out.assign(end_count=out[count_column].fillna(0).astype(float) + 1.0)
            .groupby(["season", id_column], sort=False)["end_count"].max()
        )
        previous = {
            (int(season) + 1, player): value
            for (season, player), value in end.items()
        }
        origin_key = list(zip(out["season"].astype(int), out[id_column]))
        origin = np.asarray([previous.get(item, 0.0) for item in origin_key], float)
        career = out[count_column].fillna(0).to_numpy(float)
        out[f"{role}_season_n"] = np.maximum(career - origin, 0.0)
        out[f"{role}_career_log_n"] = np.log1p(np.maximum(career, 0.0))
        out[f"{role}_season_log_n"] = np.log1p(out[f"{role}_season_n"])
    return out


def metric_gain(target: np.ndarray, base: np.ndarray, direction: np.ndarray, alpha: float) -> float:
    candidate = np.clip(base + alpha * direction, *CLIP)
    return float(bss(target, candidate) - bss(target, base))


def masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    month = frame["game_month"].to_numpy(int)
    game_type = frame["game_type"].astype(str).to_numpy()
    return {
        "all": np.ones(len(frame), dtype=bool),
        "H1": month <= 6,
        "H2": month >= 7,
        "R": game_type == "R",
        "F": game_type == "F",
    }


def main() -> None:
    columns = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_hand", "batter_hand", "asof_pitcher_n", "asof_batter_n",
        "control_success",
    ]
    all_rows = pd.read_csv(
        ROOT / "data/train.csv", usecols=columns, encoding="utf-8-sig",
        low_memory=False,
    )
    all_rows = add_season_exposure(all_rows)
    positions = np.concatenate([
        np.flatnonzero(all_rows["season"].to_numpy(int) == season)
        for season in (2023, 2024)
    ])
    rows = all_rows.iloc[positions].reset_index(drop=True)
    with np.load(ROOT / "outputs/v58_oof_predictions.npz") as archive:
        target = archive["target"].astype(float)
        base = archive["blended"].astype(float)
        season = archive["season"].astype(int)
    if len(rows) != len(target) or not np.array_equal(
        rows["season"].to_numpy(int), season,
    ):
        raise ValueError("v58 OOF and training rows are not aligned")
    if not np.array_equal(rows["control_success"].to_numpy(float), target):
        raise ValueError("v58 OOF target mismatch")

    month = rows["game_month"].to_numpy(int)
    splits = [
        Split("2023_to_2024", season == 2023, season == 2024),
        Split("2023_H1_to_H2", (season == 2023) & (month <= 6), (season == 2023) & (month >= 7)),
        Split("2024_H1_to_H2", (season == 2024) & (month <= 6), (season == 2024) & (month >= 7)),
    ]
    category_specs: list[tuple[str, tuple[str, ...], tuple[str, ...] | None, float]] = []
    for shrinkage in (500.0, 1500.0, 5000.0, 20000.0):
        category_specs.extend([
            (f"pitcher_main_k{int(shrinkage)}", ("pitcher_id",), None, shrinkage),
            (f"batter_main_k{int(shrinkage)}", ("batter_id",), None, shrinkage),
        ])
    for shrinkage in (250.0, 750.0, 1500.0, 3000.0):
        category_specs.extend([
            (
                f"pitcher_by_batter_hand_diff_k{int(shrinkage)}",
                ("pitcher_id", "batter_hand"), ("pitcher_id",), shrinkage,
            ),
            (
                f"batter_by_pitcher_hand_diff_k{int(shrinkage)}",
                ("batter_id", "pitcher_hand"), ("batter_id",), shrinkage,
            ),
        ])
    numeric_specs = []
    for column in (
        "batter_career_log_n", "pitcher_career_log_n",
        "batter_season_log_n", "pitcher_season_log_n",
    ):
        for bins in (6, 10, 16):
            for shrinkage in (2000.0, 10000.0, 40000.0):
                numeric_specs.append((f"{column}_q{bins}_k{int(shrinkage)}", column, bins, shrinkage))

    alphas = (0.5, 1.0, 2.0, 4.0)
    results: dict[str, dict] = {}
    for split in splits:
        source = rows.loc[split.source].reset_index(drop=True)
        validation = rows.loc[split.validation].reset_index(drop=True)
        source_residual = target[split.source] - base[split.source]
        validation_target = target[split.validation]
        validation_base = base[split.validation]
        segment_masks = masks(validation)
        split_results = {}
        for name, columns_, parent, shrinkage in category_specs:
            direction = frozen_category_direction(
                source, validation, source_residual, columns_, shrinkage, parent,
            )
            split_results[name] = {
                str(alpha): {
                    segment: metric_gain(
                        validation_target[mask], validation_base[mask], direction[mask], alpha,
                    )
                    for segment, mask in segment_masks.items() if mask.any()
                }
                for alpha in alphas
            }
        for name, column, bins, shrinkage in numeric_specs:
            direction = frozen_numeric_direction(
                source[column].to_numpy(float), validation[column].to_numpy(float),
                source_residual, bins, shrinkage,
            )
            split_results[name] = {
                str(alpha): {
                    segment: metric_gain(
                        validation_target[mask], validation_base[mask], direction[mask], alpha,
                    )
                    for segment, mask in segment_masks.items() if mask.any()
                }
                for alpha in alphas
            }
        results[split.name] = split_results

    rankings = []
    names = list(results[splits[0].name])
    for name in names:
        for alpha in alphas:
            gains = {
                split.name: results[split.name][name][str(alpha)]["all"]
                for split in splits
            }
            primary = results["2023_to_2024"][name][str(alpha)]
            rankings.append({
                "name": name,
                "alpha": alpha,
                "transfer_gains": gains,
                "primary_segments": primary,
                "min_transfer_gain": float(min(gains.values())),
                "mean_transfer_gain": float(np.mean(list(gains.values()))),
                "primary_gain": float(gains["2023_to_2024"]),
            })
    rankings.sort(
        key=lambda item: (item["min_transfer_gain"], item["mean_transfer_gain"], item["primary_gain"]),
        reverse=True,
    )
    report = {
        "baseline": "v58_public_feedback_counterstep",
        "method": "frozen source-residual lookups; no validation/test aggregation",
        "top_ranked": rankings[:40],
        "all_results": results,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    path = ROOT / "research/v59_player_structure.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["top_ranked"][:20], indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
