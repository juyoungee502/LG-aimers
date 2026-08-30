"""Audit an independent, leakage-safe pitcher pressure-profile residual.

Pitcher behaviour under pressure is estimated only from seasons before the
rows being represented.  A strongly regularised linear residual model is then
fit on one past OOF block and evaluated on the next block.  The script is a
research gate and does not use public fitted models or predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from research_v65_hierarchical_context_lookup import cluster_bootstrap, gain


ROOT = Path(__file__).resolve().parent
CONDITIONS = (
    "high_li", "extreme_li", "traffic", "risp", "late", "close", "behind",
    "three_ball", "two_strike", "full_count", "compound_pressure",
)
ALPHAS = (100.0, 300.0, 500.0, 1000.0)
RIDGES = (100.0, 1000.0, 10000.0)
SCALES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
CLIP = (0.005, 0.995)


def condition_frame(raw: pd.DataFrame) -> pd.DataFrame:
    li = pd.to_numeric(raw["li"], errors="coerce").fillna(0.0)
    inning = pd.to_numeric(raw["inning"], errors="coerce").fillna(0.0)
    score = pd.to_numeric(raw["score_diff_pitcher_team"], errors="coerce").fillna(0.0)
    balls = pd.to_numeric(raw["balls_before"], errors="coerce").fillna(0.0)
    strikes = pd.to_numeric(raw["strikes_before"], errors="coerce").fillna(0.0)
    runners = pd.to_numeric(raw["num_runners_on"], errors="coerce").fillna(0.0)
    risp = raw["runner_on_2b"].fillna(0).astype(bool) | raw["runner_on_3b"].fillna(0).astype(bool)
    return pd.DataFrame({
        "high_li": li.ge(1.5),
        "extreme_li": li.ge(3.0),
        "traffic": runners.gt(0),
        "risp": risp,
        "late": inning.ge(7),
        "close": score.abs().le(1),
        "behind": score.lt(0),
        "three_ball": balls.eq(3),
        "two_strike": strikes.eq(2),
        "full_count": balls.eq(3) & strikes.eq(2),
        "compound_pressure": li.ge(1.5) & (risp | balls.eq(3)) & score.abs().le(2),
    }, index=raw.index).astype(float)


def build_profile(history: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, dict[str, float]]:
    h = history.reset_index(drop=True)
    y = h["control_success"].to_numpy(float)
    cond = condition_frame(h)
    pitcher = h["pitcher_id"].astype(str)
    global_rate = float(y.mean())
    overall = pd.DataFrame({"pitcher_id": pitcher, "target": y}).groupby(
        "pitcher_id", sort=False,
    )["target"].agg(["sum", "count"])
    overall_rate = (overall["sum"] + alpha * global_rate) / (overall["count"] + alpha)
    profile = pd.DataFrame(index=overall.index)
    profile["history_log_n"] = np.log1p(overall["count"])
    profile["overall_centered"] = overall_rate - global_rate
    league: dict[str, float] = {}
    for name in CONDITIONS:
        mask = cond[name].to_numpy(bool)
        context_rate = float(y[mask].mean()) if mask.any() else global_rate
        league[name] = context_rate - global_rate
        aggregate = pd.DataFrame({
            "pitcher_id": pitcher[mask].to_numpy(), "target": y[mask],
        }).groupby("pitcher_id", sort=False)["target"].agg(["sum", "count"])
        sums = aggregate["sum"].reindex(profile.index).fillna(0.0)
        counts = aggregate["count"].reindex(profile.index).fillna(0.0)
        rate = (sums + alpha * overall_rate) / (counts + alpha)
        profile[f"{name}_effect"] = rate - overall_rate
        profile[f"{name}_reliability"] = counts / (counts + alpha)
    return profile.reset_index(), league


def attach_profile(rows: pd.DataFrame, history: pd.DataFrame, alpha: float) -> pd.DataFrame:
    profile, league = build_profile(history, alpha)
    cond = condition_frame(rows.reset_index(drop=True))
    result = pd.DataFrame({"pitcher_id": rows["pitcher_id"].astype(str).to_numpy()}).merge(
        profile, on="pitcher_id", how="left", sort=False,
    ).drop(columns="pitcher_id")
    result["history_log_n"] = result["history_log_n"].fillna(0.0)
    result["overall_centered"] = result["overall_centered"].fillna(0.0)
    for name in CONDITIONS:
        effect = result[f"{name}_effect"].fillna(league[name])
        reliability = result[f"{name}_reliability"].fillna(0.0)
        active = cond[name].to_numpy(float)
        result[f"{name}_active_effect"] = effect * active
        result[f"{name}_active_reliability"] = reliability * active
        result[f"{name}_effect"] = effect
        result[f"{name}_reliability"] = reliability
        result[f"{name}_active"] = active
    result["game_type_f"] = rows["game_type"].astype(str).eq("F").to_numpy(float)
    result["base_centered"] = rows["baseline"].to_numpy(float) - 0.5
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_predict(
    x_train: pd.DataFrame,
    residual: np.ndarray,
    x_valid: pd.DataFrame,
    ridge: float,
) -> np.ndarray:
    scaler = StandardScaler()
    train_z = scaler.fit_transform(x_train)
    valid_z = scaler.transform(x_valid)
    model = Ridge(alpha=ridge, fit_intercept=False)
    model.fit(train_z, residual)
    return model.predict(valid_z)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=3000)
    args = parser.parse_args()
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        y = archive["target"].astype(float)
        base = archive["blended"].astype(float)
        season = archive["season"].astype(int)
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(y) or not np.array_equal(rows["control_success"].to_numpy(float), y):
        raise ValueError("v64 OOF rows are not aligned")
    rows["baseline"] = base
    positions23 = np.flatnonzero(season == 2023)
    split23 = len(positions23) // 2
    fold_rows = [
        {
            "name": "2023_second_half",
            "train": rows.iloc[positions23[:split23]].copy(),
            "valid": rows.iloc[positions23[split23:]].copy(),
            "history_year": 2023,
        },
        {
            "name": "2024_forward",
            "train": rows.loc[rows["season"].eq(2023)].copy(),
            "valid": rows.loc[rows["season"].eq(2024)].copy(),
            "history_year": 2024,
        },
    ]
    predictions: dict[tuple[float, float, str], np.ndarray] = {}
    for alpha in ALPHAS:
        for fold in fold_rows:
            history = raw.loc[raw["season"].lt(fold["history_year"])]
            train_x = attach_profile(fold["train"], history, alpha)
            valid_x = attach_profile(fold["valid"], history, alpha)
            residual = fold["train"]["control_success"].to_numpy(float) - fold["train"]["baseline"].to_numpy(float)
            for ridge in RIDGES:
                predictions[(alpha, ridge, fold["name"])] = fit_predict(
                    train_x, residual, valid_x, ridge,
                )

    candidates: list[dict[str, object]] = []
    for alpha in ALPHAS:
        for ridge in RIDGES:
            for r_scale in SCALES:
                for f_scale in SCALES:
                    evaluations = []
                    for fold in fold_rows:
                        valid = fold["valid"]
                        target = valid["control_success"].to_numpy(float)
                        baseline = valid["baseline"].to_numpy(float)
                        regular = valid["game_type"].astype(str).eq("R").to_numpy()
                        scale = np.where(regular, r_scale, f_scale)
                        candidate = np.clip(
                            baseline + scale * predictions[(alpha, ridge, fold["name"])], *CLIP,
                        )
                        halves = np.array_split(np.arange(len(valid)), 2)
                        groups = {
                            label: gain(target[mask], baseline[mask], candidate[mask])
                            for label, mask in (("R", regular), ("F", ~regular))
                        }
                        evaluations.append({
                            "fold": fold["name"],
                            "gain": gain(target, baseline, candidate),
                            "half_gains": [
                                gain(target[index], baseline[index], candidate[index])
                                for index in halves
                            ],
                            "group_gains": groups,
                            "mean_absolute_change": float(np.mean(np.abs(candidate - baseline))),
                        })
                    preliminary = bool(
                        min(item["gain"] for item in evaluations) > 0.0
                        and min(v for item in evaluations for v in item["half_gains"]) >= 0.0
                        and min(v for item in evaluations for v in item["group_gains"].values()) >= 0.0
                    )
                    gains = [item["gain"] for item in evaluations]
                    candidates.append({
                        "alpha": alpha, "ridge": ridge,
                        "r_scale": r_scale, "f_scale": f_scale,
                        "evaluations": evaluations,
                        "preliminary_gate": preliminary,
                        "min_gain": float(min(gains)),
                        "mean_gain": float(np.mean(gains)),
                    })
    candidates.sort(
        key=lambda item: (item["preliminary_gate"], item["min_gain"], item["mean_gain"]),
        reverse=True,
    )
    best = candidates[0]
    bootstraps = {}
    for fold in fold_rows:
        valid = fold["valid"]
        target = valid["control_success"].to_numpy(float)
        baseline = valid["baseline"].to_numpy(float)
        regular = valid["game_type"].astype(str).eq("R").to_numpy()
        scale = np.where(regular, best["r_scale"], best["f_scale"])
        candidate = np.clip(
            baseline + scale * predictions[(best["alpha"], best["ridge"], fold["name"])], *CLIP,
        )
        bootstraps[fold["name"]] = cluster_bootstrap(
            target, baseline, candidate, valid["pitcher_id"].to_numpy(),
            args.bootstrap, 651146 + int(valid["season"].iloc[-1]),
        )
    strict = bool(
        best["preliminary_gate"]
        and min(item["ci_low"] for item in bootstraps.values()) > 0.0
    )
    report = {
        "baseline": "v64_public_method_transfer",
        "method": "independent prior-season pitcher pressure-profile Ridge residual",
        "best": best,
        "bootstrap": bootstraps,
        "strict_gate": strict,
        "selected": best if strict else None,
        "top_candidates": candidates[:20],
        "rules": {
            "official_data_only": True,
            "external_model_or_prediction_used": False,
            "forbidden_2025_trackman_used": False,
            "test_row_aggregation_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    (ROOT / "research").mkdir(exist_ok=True)
    path = ROOT / "research/v65_psych_profile_ridge.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
