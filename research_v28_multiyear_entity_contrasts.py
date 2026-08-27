"""Independent multi-year audit of entity residual levels and context contrasts.

The experiment intentionally selects no weights from public leaderboard scores.
Directions are screened on v11 rolling OOF predictions from 2020--2024, then the
same frozen hyperparameters are audited as additions to the public-best v23 OOF.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CLIP = (.005, .995)
LEVEL_K = (5_000., 20_000., 50_000., 100_000.)
CONTRAST_K = (100., 300., 1_000., 2_000., 5_000., 10_000.)
SCALES = (.05, .10, .20, .35, .50, .75, 1., 1.5, 2., 3.)
KINDS = {
    "pitcher_level": ("level", "pitcher_id", None),
    "batter_level": ("level", "batter_id", None),
    "pitcher_hand_contrast": ("contrast", "pitcher_id", "same_hand"),
    "pitcher_two_strike_contrast": (
        "contrast", "pitcher_id", "two_strike",
    ),
    "pitcher_runner_contrast": ("contrast", "pitcher_id", "runner_present"),
}


def bss_gain(target, base, correction):
    target = np.asarray(target, dtype=float)
    base = np.asarray(base, dtype=float)
    candidate = np.clip(base + correction, *CLIP)
    reference = float(target.mean() * (1. - target.mean()))
    if reference <= 0:
        return 0.
    return float(
        100_000. * (np.mean((target - base) ** 2)
                    - np.mean((target - candidate) ** 2)) / reference
    )


def add_contexts(frame):
    frame = frame.copy()
    frame["same_hand"] = frame["pitcher_hand"].eq(frame["batter_hand"]).astype(np.int8)
    frame["two_strike"] = frame["strikes_before"].eq(2).astype(np.int8)
    frame["runner_present"] = frame["num_runners_on"].gt(0).astype(np.int8)
    return frame


def make_year_data():
    columns = [
        "season", "game_type", "pitcher_id", "batter_id", "pitcher_hand",
        "batter_hand", "strikes_before", "balls_before", "num_runners_on",
        "control_success",
    ]
    raw = add_contexts(pd.read_csv(
        ROOT / "data/train.csv", usecols=columns, low_memory=False,
    ))
    archives = (
        ROOT / "research/rolling_2020_2021/v11_oof_predictions.npz",
        ROOT / "research/rolling_2022_2024/v11_oof_predictions.npz",
    )
    result = {}
    for path in archives:
        with np.load(path) as loaded:
            seasons = loaded["season"].astype(int)
            target = loaded["target"].astype(float)
            base = loaded["blended"].astype(float)
        for year in np.unique(seasons):
            mask = seasons == year
            frame = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
            if len(frame) != int(mask.sum()):
                raise ValueError(f"row mismatch for {year}: {len(frame)} != {mask.sum()}")
            y = target[mask]
            if not np.array_equal(frame["control_success"].to_numpy(float), y):
                raise ValueError(f"target order mismatch for {year}")
            result[int(year)] = {
                "frame": frame.drop(columns="control_success"),
                "target": y,
                "base": base[mask],
            }
    return result


def combine_years(year_data, years):
    frames, targets, bases = [], [], []
    for year in years:
        frames.append(year_data[year]["frame"])
        targets.append(year_data[year]["target"])
        bases.append(year_data[year]["base"])
    return {
        "frame": pd.concat(frames, ignore_index=True),
        "target": np.concatenate(targets),
        "base": np.concatenate(bases),
    }


def fit_sufficient(source_frame, source_residual, kind, scope):
    mode, entity, context = KINDS[kind]
    active = np.ones(len(source_frame), dtype=bool)
    if scope != "all":
        active &= source_frame["game_type"].eq(scope).to_numpy()
    work = source_frame.loc[active, [entity] + ([context] if context else [])].copy()
    work["residual"] = np.asarray(source_residual)[active]
    if mode == "level":
        return work.groupby(entity, observed=True)["residual"].agg(
            residual_sum="sum", residual_n="count",
        )
    grouped = work.groupby([entity, context], observed=True)["residual"].agg(
        residual_sum="sum", residual_n="count",
    ).reset_index()
    total = grouped.pivot(index=entity, columns=context, values="residual_sum")
    count = grouped.pivot(index=entity, columns=context, values="residual_n")
    for value in (0, 1):
        if value not in total:
            total[value] = np.nan
            count[value] = np.nan
    table = pd.DataFrame({
        "sum0": total[0], "n0": count[0],
        "sum1": total[1], "n1": count[1],
    }).dropna()
    return table


def apply_sufficient(query, sufficient, kind, scope, shrink):
    mode, entity, context = KINDS[kind]
    key = query[entity]
    if mode == "level":
        effect = sufficient["residual_sum"] / (sufficient["residual_n"] + shrink)
        direction = key.map(effect).fillna(0.).to_numpy(float, copy=True)
    else:
        mean0 = sufficient["sum0"] / sufficient["n0"]
        mean1 = sufficient["sum1"] / sufficient["n1"]
        effective_n = (
            sufficient["n0"] * sufficient["n1"]
            / (sufficient["n0"] + sufficient["n1"])
        )
        effect = (mean1 - mean0) * effective_n / (effective_n + shrink)
        mapped = key.map(effect).fillna(0.).to_numpy(float, copy=True)
        sign = np.where(query[context].to_numpy() == 1, .5, -.5)
        direction = mapped * sign
    if scope != "all":
        direction[~query["game_type"].eq(scope).to_numpy()] = 0.
    return direction


def segment_gains(valid, correction):
    frame, target, base = valid["frame"], valid["target"], valid["base"]
    masks = {"all": np.ones(len(frame), dtype=bool)}
    for game_type in ("R", "F"):
        mask = frame["game_type"].eq(game_type).to_numpy()
        if mask.any():
            masks[game_type] = mask
    return {
        label: bss_gain(target[mask], base[mask], correction[mask])
        for label, mask in masks.items()
    }


def rolling_screen(year_data):
    transfer_specs = {
        "20_to_21": ((2020,), (2021,)),
        "21_to_22": ((2021,), (2022,)),
        "22_to_23": ((2022,), (2023,)),
        "23_to_24": ((2023,), (2024,)),
        "20_21_to_22": ((2020, 2021), (2022,)),
        "21_22_to_23": ((2021, 2022), (2023,)),
        "22_23_to_24": ((2022, 2023), (2024,)),
    }
    transfers = {
        name: (combine_years(year_data, source), combine_years(year_data, valid))
        for name, (source, valid) in transfer_specs.items()
    }
    candidates = []
    for kind, (mode, _entity, _context) in KINDS.items():
        shrinks = LEVEL_K if mode == "level" else CONTRAST_K
        for scope in ("all", "R", "F"):
            sufficient = {
                label: fit_sufficient(
                    source["frame"], source["target"] - source["base"], kind, scope,
                )
                for label, (source, _valid) in transfers.items()
            }
            for shrink in shrinks:
                directions = {
                    label: apply_sufficient(valid["frame"], sufficient[label], kind,
                                            scope, shrink)
                    for label, (_source, valid) in transfers.items()
                }
                for scale in SCALES:
                    gains = {
                        label: segment_gains(valid, scale * directions[label])
                        for label, (_source, valid) in transfers.items()
                    }
                    recent = ("21_to_22", "22_to_23", "23_to_24")
                    pooled = ("20_21_to_22", "21_22_to_23", "22_23_to_24")
                    active_segment = ("all", "R", "F") if scope == "all" else (scope,)
                    recent_detail = [
                        gains[label][segment]
                        for label in recent for segment in active_segment
                        if segment in gains[label]
                    ]
                    row = {
                        "kind": kind, "scope": scope, "shrink": shrink,
                        "scale": scale, "gains": gains,
                        "min_recent_all": min(gains[x]["all"] for x in recent),
                        "min_pooled_all": min(gains[x]["all"] for x in pooled),
                        "min_recent_active_segment": min(recent_detail),
                        "mean_recent_all": float(np.mean([gains[x]["all"] for x in recent])),
                    }
                    candidates.append(row)
    candidates.sort(key=lambda x: (
        x["min_recent_active_segment"], x["min_pooled_all"], x["mean_recent_all"],
    ), reverse=True)
    return candidates


def make_v23_data():
    raw = add_contexts(pd.read_csv(
        ROOT / "data/train.csv",
        usecols=[
            "season", "game_type", "pitcher_id", "batter_id", "pitcher_hand",
            "batter_hand", "strikes_before", "balls_before", "num_runners_on",
            "control_success",
        ], low_memory=False,
    ))
    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as loaded:
        seasons = loaded["season"].astype(int)
        target = loaded["target"].astype(float)
        base = loaded["blended"].astype(float)
    result = {}
    for year in (2023, 2024):
        mask = seasons == year
        frame = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        if not np.array_equal(frame["control_success"].to_numpy(float), target[mask]):
            raise ValueError(f"v23 target order mismatch for {year}")
        result[year] = {
            "frame": frame.drop(columns="control_success"),
            "target": target[mask], "base": base[mask],
        }
    return result


def section(item, start, stop):
    return {
        "frame": item["frame"].iloc[start:stop].reset_index(drop=True),
        "target": item["target"][start:stop],
        "base": item["base"][start:stop],
    }


def v23_audit(candidates):
    years = make_v23_data()
    n23, n24 = len(years[2023]["frame"]), len(years[2024]["frame"])
    q24 = np.linspace(0, n24, 5, dtype=int)
    source23 = years[2023]
    source23h1 = section(years[2023], 0, n23 // 2)
    source24h1 = section(years[2024], 0, n24 // 2)
    blocks = {
        "23h1_to_23h2": (source23h1, section(years[2023], n23 // 2, n23)),
        "23_to_24": (source23, years[2024]),
        "24h1_to_24h2": (source24h1, section(years[2024], n24 // 2, n24)),
    }
    for quarter in range(4):
        blocks[f"23_to_24q{quarter + 1}"] = (
            source23, section(years[2024], q24[quarter], q24[quarter + 1]),
        )

    # Include all structurally safe candidates plus a bounded diagnostic frontier.
    selected = [x for x in candidates if (
        x["min_recent_active_segment"] >= 0.
        and x["min_pooled_all"] >= 0.
    )]
    selected_keys = {
        (x["kind"], x["scope"], x["shrink"], x["scale"]) for x in selected
    }
    for row in candidates[:120]:
        key = (row["kind"], row["scope"], row["shrink"], row["scale"])
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)

    stats_cache = {}
    direction_cache = {}
    audited = []
    for candidate in selected:
        kind, scope = candidate["kind"], candidate["scope"]
        shrink, scale = candidate["shrink"], candidate["scale"]
        gains = {}
        for label, (source, valid) in blocks.items():
            stats_key = (kind, scope, label)
            if stats_key not in stats_cache:
                stats_cache[stats_key] = fit_sufficient(
                    source["frame"], source["target"] - source["base"], kind, scope,
                )
            direction_key = (*stats_key, shrink)
            if direction_key not in direction_cache:
                direction_cache[direction_key] = apply_sufficient(
                    valid["frame"], stats_cache[stats_key], kind, scope, shrink,
                )
            gains[label] = segment_gains(
                valid, scale * direction_cache[direction_key],
            )
        active_segment = ("all", "R", "F") if scope == "all" else (scope,)
        detail = [
            values[segment]
            for values in gains.values() for segment in active_segment
            if segment in values
        ]
        audited.append({
            "kind": kind, "scope": scope, "shrink": shrink, "scale": scale,
            "rolling_min_recent_active_segment": candidate["min_recent_active_segment"],
            "rolling_min_pooled_all": candidate["min_pooled_all"],
            "rolling_mean_recent_all": candidate["mean_recent_all"],
            "v23_gains": gains,
            "v23_min_active_segment": min(detail),
            "v23_mean_all": float(np.mean([x["all"] for x in gains.values()])),
        })
    audited.sort(key=lambda x: (
        x["v23_min_active_segment"], x["rolling_min_recent_active_segment"],
        x["v23_mean_all"],
    ), reverse=True)
    return audited


def main():
    year_data = make_year_data()
    rolling = rolling_screen(year_data)
    audited = v23_audit(rolling)
    safe = [x for x in audited if (
        x["rolling_min_recent_active_segment"] >= 0.
        and x["rolling_min_pooled_all"] >= 0.
        and x["v23_min_active_segment"] >= 0.
    )]
    report = {
        "method": "multi-year rolling residual entity/context audit",
        "leaderboard_weights_used": False,
        "rolling_candidate_count": len(rolling),
        "rolling_safe_count": sum(
            x["min_recent_active_segment"] >= 0. and x["min_pooled_all"] >= 0.
            for x in rolling
        ),
        "v23_audited_count": len(audited),
        "fully_safe_count": len(safe),
        "fully_safe": safe[:100],
        "top_rolling": rolling[:100],
        "top_v23_audit": audited[:100],
    }
    output = ROOT / "research/v28_multiyear_entity_contrasts.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "rolling_candidate_count": report["rolling_candidate_count"],
        "rolling_safe_count": report["rolling_safe_count"],
        "v23_audited_count": report["v23_audited_count"],
        "fully_safe_count": report["fully_safe_count"],
        "fully_safe": safe[:10],
        "top_rolling": rolling[:5],
        "top_v23_audit": audited[:5],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
