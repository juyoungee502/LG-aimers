"""Audit public-reported pitcher/hand conditional effects independently.

Conditional target deviations are fit on an earlier block only. Hyperparameters
are screened across 2020->2021 through 2023->2024 rolling OOF, then applied
unchanged on top of v23 for a separate robustness audit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
HAND_K = (50., 110., 200., 500., 1_000.)
PRESSURE_K = (100., 220., 500., 1_000., 2_000.)
SCALES = (0., .025, .05, .10, .20, .35, .50, .75, 1.)


def bss_gain(y, base, correction):
    ref = float(y.mean() * (1. - y.mean()))
    candidate = np.clip(base + correction, .005, .995)
    return float(100_000. * (
        np.mean((y - base) ** 2) - np.mean((y - candidate) ** 2)
    ) / ref)


def prepare(frame):
    frame = frame.copy()
    balls = frame["balls_before"].to_numpy()
    strikes = frame["strikes_before"].to_numpy()
    frame["pressure_state"] = np.where(
        (balls == 3) & (strikes == 2), 2,
        np.where((balls == 3) | (strikes == 2), 1, 0),
    ).astype(np.int8)
    return frame


def fit_stats(frame, target, child_keys, parent_keys):
    source = frame.loc[frame["game_type"].eq("R"), [*child_keys]].copy()
    source["target"] = target[frame["game_type"].eq("R").to_numpy()]
    parents = source.groupby(list(parent_keys), observed=True)["target"].agg(
        parent_sum="sum", parent_n="count",
    ).reset_index()
    children = source.groupby(list(child_keys), observed=True)["target"].agg(
        child_sum="sum", child_n="count",
    ).reset_index()
    return children.merge(parents, on=list(parent_keys), how="left")


def apply_stats(query, stats, child_keys, shrink):
    table = stats.copy()
    parent_rate = table["parent_sum"] / table["parent_n"]
    table["effect"] = (
        table["child_sum"] - table["child_n"] * parent_rate
    ) / (table["child_n"] + shrink)
    work = query[list(child_keys)].copy()
    work["_order"] = np.arange(len(work))
    work = work.merge(
        table[[*child_keys, "effect"]], on=list(child_keys), how="left", sort=False,
    ).sort_values("_order")
    direction = work["effect"].fillna(0.).to_numpy(float, copy=True)
    direction[~query["game_type"].eq("R").to_numpy()] = 0.
    return direction


def raw_years():
    raw = prepare(pd.read_csv(
        ROOT / "data/train.csv",
        usecols=[
            "season", "game_type", "pitcher_id", "batter_hand",
            "balls_before", "strikes_before", "control_success",
        ], low_memory=False,
    ))
    return {
        year: raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for year in range(2020, 2025)
    }


def rolling_data(raw):
    result = {}
    for path in (
        ROOT / "research/rolling_2020_2021/v11_oof_predictions.npz",
        ROOT / "research/rolling_2022_2024/v11_oof_predictions.npz",
    ):
        with np.load(path) as z:
            for year in np.unique(z["season"]):
                mask = z["season"] == year
                frame = raw[int(year)]
                y = z["target"][mask].astype(float)
                if not np.array_equal(frame["control_success"].to_numpy(float), y):
                    raise ValueError(f"target mismatch: {year}")
                result[int(year)] = {
                    "frame": frame.drop(columns="control_success"),
                    "target": y, "base": z["blended"][mask].astype(float),
                }
    return result


def blocks_for_rolling(data):
    return {
        f"{year}_to_{year + 1}": (data[year], data[year + 1])
        for year in range(2020, 2024)
    }


def fit_block_directions(blocks):
    fitted = {}
    for label, (source, valid) in blocks.items():
        hand_stats = fit_stats(
            source["frame"], source["target"],
            ("pitcher_id", "batter_hand"), ("pitcher_id",),
        )
        pressure_stats = fit_stats(
            source["frame"], source["target"],
            ("pitcher_id", "batter_hand", "pressure_state"),
            ("pitcher_id", "batter_hand"),
        )
        fitted[label] = {
            "valid": valid,
            "hand": {
                k: apply_stats(valid["frame"], hand_stats,
                               ("pitcher_id", "batter_hand"), k)
                for k in HAND_K
            },
            "pressure": {
                k: apply_stats(valid["frame"], pressure_stats,
                               ("pitcher_id", "batter_hand", "pressure_state"), k)
                for k in PRESSURE_K
            },
        }
    return fitted


def segment_values(valid, correction):
    frame, y, base = valid["frame"], valid["target"], valid["base"]
    regular = np.flatnonzero(frame["game_type"].eq("R").to_numpy())
    cuts = np.linspace(0, len(regular), 5, dtype=int)
    values = {"all": bss_gain(y, base, correction)}
    values["R"] = bss_gain(y[regular], base[regular], correction[regular])
    for part in range(4):
        rows = regular[cuts[part]:cuts[part + 1]]
        values[f"Rq{part + 1}"] = bss_gain(y[rows], base[rows], correction[rows])
    return values


def segment_masks(valid):
    frame = valid["frame"]
    regular = np.flatnonzero(frame["game_type"].eq("R").to_numpy())
    cuts = np.linspace(0, len(regular), 5, dtype=int)
    result = {
        "all": np.ones(len(frame), dtype=bool),
        "R": frame["game_type"].eq("R").to_numpy(),
    }
    positions = np.arange(len(frame))
    for part in range(4):
        result[f"Rq{part + 1}"] = np.isin(
            positions, regular[cuts[part]:cuts[part + 1]], assume_unique=True,
        )
    return result


def curve_coefficients(valid, hand, pressure):
    y, base = valid["target"], valid["base"]
    curves = {}
    for name, active in segment_masks(valid).items():
        yy, bb = y[active], base[active]
        hh, pp = hand[active], pressure[active]
        residual = yy - bb
        ref = float(yy.mean() * (1. - yy.mean()))
        curves[name] = (
            200_000. * float(np.mean(residual * hh)) / ref,
            200_000. * float(np.mean(residual * pp)) / ref,
            100_000. * float(np.mean(hh * hh)) / ref,
            100_000. * float(np.mean(pp * pp)) / ref,
            100_000. * float(np.mean(hh * pp)) / ref,
        )
    return curves


def gains_from_curves(curves, hand_scale, pressure_scale):
    return {
        name: (
            lh * hand_scale + lp * pressure_scale
            - qh * hand_scale**2 - qp * pressure_scale**2
            - 2. * qhp * hand_scale * pressure_scale
        )
        for name, (lh, lp, qh, qp, qhp) in curves.items()
    }


def screen(fitted):
    candidates = []
    for hand_k in HAND_K:
        for pressure_k in PRESSURE_K:
            curves = {
                label: curve_coefficients(
                    block["valid"], block["hand"][hand_k],
                    block["pressure"][pressure_k],
                )
                for label, block in fitted.items()
            }
            for hand_scale in SCALES:
                for pressure_scale in SCALES:
                    if hand_scale == pressure_scale == 0.:
                        continue
                    gains = {
                        label: gains_from_curves(value, hand_scale, pressure_scale)
                        for label, value in curves.items()
                    }
                    detail = [value for row in gains.values() for value in row.values()]
                    candidates.append({
                        "hand_k": hand_k, "pressure_k": pressure_k,
                        "hand_scale": hand_scale, "pressure_scale": pressure_scale,
                        "gains": gains, "min_detail": min(detail),
                        "min_year_R": min(x["R"] for x in gains.values()),
                        "mean_year_R": float(np.mean([x["R"] for x in gains.values()])),
                    })
    candidates.sort(
        key=lambda x: (x["min_detail"], x["min_year_R"], x["mean_year_R"]),
        reverse=True,
    )
    return candidates


def v23_blocks(raw):
    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as z:
        data = {}
        for year in (2023, 2024):
            mask = z["season"] == year
            frame = raw[year].drop(columns="control_success")
            data[year] = {
                "frame": frame, "target": z["target"][mask].astype(float),
                "base": z["blended"][mask].astype(float),
            }
    n23, n24 = len(data[2023]["frame"]), len(data[2024]["frame"])

    def take(item, start, stop):
        return {
            "frame": item["frame"].iloc[start:stop].reset_index(drop=True),
            "target": item["target"][start:stop], "base": item["base"][start:stop],
        }

    return {
        "23h1_to_23h2": (
            take(data[2023], 0, n23 // 2), take(data[2023], n23 // 2, n23),
        ),
        "23_to_24": (data[2023], data[2024]),
        "24h1_to_24h2": (
            take(data[2024], 0, n24 // 2), take(data[2024], n24 // 2, n24),
        ),
    }


def audit_v23(candidates, fitted):
    chosen = [x for x in candidates if x["min_detail"] >= 0.]
    chosen.extend(candidates[:100])
    seen, audited, curve_cache = set(), [], {}
    for row in chosen:
        key = (row["hand_k"], row["pressure_k"], row["hand_scale"],
               row["pressure_scale"])
        if key in seen:
            continue
        seen.add(key)
        pair = (row["hand_k"], row["pressure_k"])
        if pair not in curve_cache:
            curve_cache[pair] = {
                label: curve_coefficients(
                    block["valid"], block["hand"][row["hand_k"]],
                    block["pressure"][row["pressure_k"]],
                )
                for label, block in fitted.items()
            }
        gains = {
            label: gains_from_curves(
                curves, row["hand_scale"], row["pressure_scale"],
            )
            for label, curves in curve_cache[pair].items()
        }
        detail = [value for values in gains.values() for value in values.values()]
        audited.append({
            **{k: row[k] for k in (
                "hand_k", "pressure_k", "hand_scale", "pressure_scale",
                "min_detail", "min_year_R", "mean_year_R",
            )},
            "v23_gains": gains, "v23_min_detail": min(detail),
            "v23_mean_R": float(np.mean([x["R"] for x in gains.values()])),
        })
    audited.sort(
        key=lambda x: (x["v23_min_detail"], x["min_detail"], x["v23_mean_R"]),
        reverse=True,
    )
    return audited


def main():
    raw = raw_years()
    rolling = screen(fit_block_directions(blocks_for_rolling(rolling_data(raw))))
    audited = audit_v23(rolling, fit_block_directions(v23_blocks(raw)))
    safe = [x for x in audited if x["min_detail"] >= 0. and x["v23_min_detail"] >= 0.]
    report = {
        "leaderboard_weights_used": False,
        "rolling_safe_count": sum(x["min_detail"] >= 0. for x in rolling),
        "fully_safe_count": len(safe),
        "fully_safe": safe[:100], "top_rolling": rolling[:100],
        "top_v23": audited[:100],
    }
    output = ROOT / "research/v28_hierarchical_conditionals.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "rolling_safe_count": report["rolling_safe_count"],
        "fully_safe_count": report["fully_safe_count"],
        "fully_safe": safe[:10], "top_rolling": rolling[:5],
        "top_v23": audited[:5],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
