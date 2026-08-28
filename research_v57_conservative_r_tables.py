"""Screen one conservative R residual table at a time over the v56 anchor.

This revisits only the individually stable v25 table directions.  The failed
v26 portfolio combined many such directions aggressively; here every candidate
must stand on its own across four chronological transfers and all time slices.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)
from research_v40_failure_seed_stability import logit, sigmoid
from train_v25_temporal_portfolio import bss, segment_masks
from v25_temporal_portfolio import apply_regime, freeze_regime


ROOT = Path(__file__).resolve().parent
F_SCALE = 1.25
warnings.filterwarnings("ignore", category=PerformanceWarning)


def candidate_specs() -> list[dict]:
    one_d = json.loads(
        (ROOT / "research/v25_r_portfolio.json").read_text(encoding="utf-8")
    )["candidates"]
    pairs = json.loads(
        (ROOT / "research/v25_r_pair_transfer.json").read_text(encoding="utf-8")
    )["top"][:15]
    specs: list[dict] = []
    seen = set()
    for raw in [*one_d, *pairs]:
        if "context" in raw:
            key = ("pair", raw["column"], raw["context"], raw["bins"], raw["shrink"], raw["scale"])
            spec = {
                "type": "pair", "column": raw["column"],
                "context": raw["context"], "bins": raw["bins"],
                "shrink": raw["shrink"], "scale": raw["scale"],
            }
        else:
            key = ("one_d", raw["column"], raw["bins"], raw["shrink"], raw["scale"])
            spec = {
                "type": "one_d", "kind": raw["kind"],
                "column": raw["column"], "bins": raw["bins"],
                "shrink": raw["shrink"], "scale": raw["scale"],
            }
        if key not in seen:
            seen.add(key)
            specs.append(spec)
    return specs


def main() -> None:
    full = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = full.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    history = training_history_arrays(full, target_series)
    features_all = engineer_features(
        full, *history, global_prior=float(target_series.mean())
    )
    add_training_component_features(features_all, full)
    features_all = add_state_interactions(features_all)

    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        v38 = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        v54 = {key: archive[key] for key in archive.files}
    seasons = full["season"].to_numpy(np.int16)
    positions = np.concatenate([
        np.flatnonzero(seasons == year) for year in (2023, 2024)
    ])
    rows = full.iloc[positions].reset_index(drop=True)
    features = features_all.iloc[positions].reset_index(drop=True)
    y = v38["target"].astype(float)
    year = v38["season"].astype(int)
    if not np.allclose(target_all[positions], y):
        raise ValueError("OOF rows do not align with train.csv")

    base = v54["blended"].astype(float).copy()
    active_2024 = year == 2024
    futures_2024 = active_2024 & rows["game_type"].eq("F").to_numpy()
    base[futures_2024] = sigmoid(
        logit(v38["blended"][futures_2024].astype(float))
        + F_SCALE * (
            logit(v54["blended"][futures_2024].astype(float))
            - logit(v38["blended"][futures_2024].astype(float))
        )
    )

    regular = rows["game_type"].eq("R").to_numpy()
    indices = {
        value: np.flatnonzero(regular & (year == value))
        for value in (2023, 2024)
    }
    halves = {
        (value, half): index[:len(index) // 2] if half == 1 else index[len(index) // 2:]
        for value, index in indices.items() for half in (1, 2)
    }
    transfers = (
        ("23h1_to_23h2", halves[(2023, 1)], halves[(2023, 2)]),
        ("23_to_24h1", indices[2023], halves[(2024, 1)]),
        ("23_to_24h2", indices[2023], halves[(2024, 2)]),
        ("24h1_to_24h2", halves[(2024, 1)], halves[(2024, 2)]),
        ("2024", indices[2023], indices[2024]),
    )

    reports = []
    for index, raw_spec in enumerate(candidate_specs(), start=1):
        unit_spec = {**raw_spec, "weight": 1.0}
        frozen_directions = {}
        for label, source, valid in transfers:
            frozen = freeze_regime(
                rows.iloc[source], features.iloc[source], base[source], y[source],
                (unit_spec,), (),
            )
            frozen_directions[label] = (
                valid,
                apply_regime(
                    rows.iloc[valid], features.iloc[valid], base[valid], frozen,
                ),
            )
        for multiplier in (0.25, 0.5, 0.75, 1.0):
            spec = {**raw_spec, "weight": multiplier}
            metrics = {}
            for label, (valid, direction) in frozen_directions.items():
                candidate = np.clip(base[valid] + multiplier * direction, 0.005, 0.995)
                query = rows.iloc[valid].reset_index(drop=True)
                for name, mask in segment_masks(query, label).items():
                    metrics[name] = bss(y[valid][mask], candidate[mask]) - bss(
                        y[valid][mask], base[valid][mask]
                    )
            transfer_all = [
                metrics[f"{name}/all"] for name in (
                    "23h1_to_23h2", "23_to_24h1", "23_to_24h2",
                    "24h1_to_24h2",
                )
            ]
            quarters = [
                value for name, value in metrics.items()
                if "/q" in name
            ]
            halves = [
                value for name, value in metrics.items()
                if "/half_" in name
            ]
            months = [
                value for name, value in metrics.items()
                if "/month_" in name
            ]
            reports.append({
                "spec": spec,
                "gain_2024_R": float(metrics["2024/all"]),
                "min_transfer_all": float(min(transfer_all)),
                "min_quarter": float(min(quarters)),
                "min_half": float(min(halves)),
                "min_month": float(min(months)),
                "metrics": metrics,
            })
        print(f"screened {index} candidate directions", flush=True)

    ranked = sorted(reports, key=lambda item: (
        min(item["min_transfer_all"], item["min_quarter"], item["min_half"]),
        item["gain_2024_R"], item["min_month"],
    ), reverse=True)
    safe = [item for item in ranked if (
        item["gain_2024_R"] >= 5.0
        and item["min_transfer_all"] > 0.0
        and item["min_quarter"] > 0.0
        and item["min_half"] > 0.0
    )]
    output = {
        "anchor": "v56",
        "screened": len(reports),
        "selection": (
            "One table only; >=5 R-BSS on 2024 and positive on every "
            "chronological transfer, half, and quarter."
        ),
        "safe": safe,
        "ranked": ranked[:100],
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    path = ROOT / "research/v57_conservative_r_tables.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"safe": safe[:20], "top": ranked[:20]}, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
