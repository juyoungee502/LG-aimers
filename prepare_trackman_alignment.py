"""Reliably align official main-table pitches to historical Trackman logs.

The two tables have different anonymous identifiers.  Games are linked by
season/date fingerprint, teams, pitch counts, and pre-pitch state sequences;
only games with at least 90% sequence coverage are retained.
"""
from __future__ import annotations

import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


MAJOR_TEAMS = {
    "DOO_BEA", "HAN_EAG", "KIA_TIG", "KIW_HER", "KT_WIZ", "LG_TWI",
    "LOT_GIA", "NC_DIN", "SAM_LIO", "SK_WYV", "SSG_LAN",
}


def train_games(frame):
    low = np.minimum(frame["pitcher_team_id"], frame["batter_team_id"])
    high = np.maximum(frame["pitcher_team_id"], frame["batter_team_id"])
    key = (
        frame["season"].to_numpy(np.int64) * 100_000_000
        + frame["game_month"].to_numpy(np.int64) * 10_000_000
        + frame["game_dayofweek"].to_numpy(np.int64) * 1_000_000
        + low.to_numpy(np.int64) * 100 + high.to_numpy(np.int64)
    )
    inning = frame["inning"].to_numpy()
    reset = np.r_[True, (key[1:] != key[:-1]) | (inning[1:] < inning[:-1])]
    frame = frame.copy()
    frame["game_id"] = np.cumsum(reset) - 1
    top = frame["top_bottom"].eq("T")
    frame["home"] = np.where(top, frame["pitcher_team_id"], frame["batter_team_id"])
    frame["away"] = np.where(top, frame["batter_team_id"], frame["pitcher_team_id"])
    games = frame.groupby("game_id", sort=False).agg(
        season=("season", "first"), month=("game_month", "first"),
        weekday=("game_dayofweek", "first"), home=("home", "first"),
        away=("away", "first"), size=("season", "size"),
    )
    return frame, games


def trackman_games(frame):
    top = frame["top_bottom"].eq("Top")
    frame = frame.copy()
    frame["home"] = np.where(top, frame["pitcher_team"], frame["batter_team"])
    frame["away"] = np.where(top, frame["batter_team"], frame["pitcher_team"])
    games = frame.groupby("trackman_game_id", sort=False).agg(
        season=("season", "first"), date=("game_date", "first"),
        month=("game_month", "first"), weekday=("game_dayofweek", "first"),
        home=("home", "first"), away=("away", "first"),
        size=("season", "size"),
    )
    games["date"] = pd.to_datetime(games["date"], format="mixed")
    return frame, games.sort_values(["season", "date"])


def infer_team_maps(train, trackman):
    maps = {}
    for season in sorted(train["season"].unique()):
        main = train.loc[train["season"].eq(season)].groupby("pitcher_team_id").size()
        history = trackman.loc[trackman["season"].eq(season)].groupby("pitcher_team").size()
        main_rate = main.to_numpy(float) / main.sum()
        history_rate = history.to_numpy(float) / history.sum()
        rows, columns = linear_sum_assignment(
            np.abs(main_rate[:, None] - history_rate[None, :])
        )
        maps[int(season)] = {
            int(main.index[i]): str(history.index[j]) for i, j in zip(rows, columns)
        }
    return maps


def match_games(main_games, trackman_games_frame, team_maps):
    matches = {}
    for season in sorted(main_games["season"].unique()):
        main = main_games.loc[main_games["season"].eq(season)].copy()
        track = trackman_games_frame.loc[trackman_games_frame["season"].eq(season)]
        mapping = team_maps[int(season)]
        main["home_name"] = main["home"].map(mapping)
        main["away_name"] = main["away"].map(mapping)
        for keys, group in main.groupby(
            ["month", "weekday", "home_name", "away_name"], observed=True,
        ):
            candidates = track.loc[
                track["month"].eq(keys[0]) & track["weekday"].eq(keys[1])
                & track["home"].eq(keys[2]) & track["away"].eq(keys[3])
            ]
            if candidates.empty:
                continue
            a = group["size"].to_numpy(float)
            b = candidates["size"].to_numpy(float)
            cost = np.abs(a[:, None] - b[None, :]) / np.maximum(a[:, None], b[None, :])
            rows, columns = linear_sum_assignment(cost)
            for i, j in zip(rows, columns):
                if cost[i, j] <= 0.45:
                    matches[int(group.index[i])] = str(candidates.index[j])
    return matches


def state_tokens(frame, trackman=False):
    if trackman:
        half = frame["top_bottom"].eq("Bottom").to_numpy(np.int64)
        pitcher_hand = frame["pitcher_hand"].eq("Right").to_numpy(np.int64)
        batter_hand = frame["batter_hand"].eq("Right").to_numpy(np.int64)
    else:
        half = frame["top_bottom"].eq("B").to_numpy(np.int64)
        pitcher_hand = frame["pitcher_hand"].eq(2).to_numpy(np.int64)
        batter_hand = frame["batter_hand"].eq(2).to_numpy(np.int64)
    return (((((frame["inning"].to_numpy(np.int64) * 2 + half) * 4
                + frame["balls_before"].to_numpy(np.int64)) * 3
               + frame["strikes_before"].to_numpy(np.int64)) * 3
              + frame["outs_before"].to_numpy(np.int64)) * 2
            + pitcher_hand) * 2 + batter_hand


def main():
    root = Path(__file__).resolve().parent
    train_columns = [
        "row_id", "season", "game_month", "game_dayofweek", "inning",
        "top_bottom", "game_type", "balls_before", "strikes_before",
        "outs_before", "pitcher_id", "pitcher_hand", "batter_hand",
        "pitcher_team_id", "batter_team_id",
    ]
    trackman_columns = [
        "trackman_id", "season", "game_date", "game_month", "game_dayofweek",
        "trackman_game_id", "pitch_no", "inning", "top_bottom", "balls_before",
        "strikes_before", "outs_before", "pitcher_trackman_id", "pitcher_hand",
        "batter_hand", "pitcher_team", "batter_team",
    ]
    train = pd.read_csv(
        root / "data" / "train.csv", usecols=train_columns,
        encoding="utf-8-sig", low_memory=False,
    )
    train = train.loc[train["game_type"].eq("R")].drop(columns="game_type").reset_index(drop=True)
    trackman = pd.read_csv(
        root / "data" / "trackman_history.csv", usecols=trackman_columns,
        encoding="utf-8-sig", low_memory=False,
    )
    trackman = trackman.loc[
        trackman["pitcher_team"].isin(MAJOR_TEAMS)
        & trackman["batter_team"].isin(MAJOR_TEAMS)
    ].reset_index(drop=True)
    train, main_games = train_games(train)
    trackman, history_games = trackman_games(trackman)
    team_maps = infer_team_maps(train, trackman)
    matches = match_games(main_games, history_games, team_maps)
    track_groups = {
        str(key): group.sort_values("pitch_no")
        for key, group in trackman.groupby("trackman_game_id", sort=False)
    }
    aligned_main, aligned_trackman, details = [], [], []
    for main_id, history_id in matches.items():
        main = train.loc[train["game_id"].eq(main_id)]
        history = track_groups[history_id]
        a, b = state_tokens(main), state_tokens(history, trackman=True)
        direct = float(np.mean(a == b)) if len(a) == len(b) else 0.0
        if len(a) == len(b) and direct >= 0.995:
            blocks, method = [(0, 0, len(a))], "direct"
        else:
            blocks = [
                (block.a, block.b, block.size)
                for block in SequenceMatcher(
                    None, a.tolist(), b.tolist(), autojunk=False,
                ).get_matching_blocks() if block.size
            ]
            method = "sequence"
        matched = sum(size for _, _, size in blocks)
        coverage = matched / max(1, min(len(main), len(history)))
        accepted = coverage >= 0.90
        if accepted:
            main_ids = main["row_id"].to_numpy()
            track_ids = history["trackman_id"].to_numpy()
            for i, j, size in blocks:
                aligned_main.append(main_ids[i:i + size])
                aligned_trackman.append(track_ids[j:j + size])
        details.append({
            "main_game_id": int(main_id), "trackman_game_id": history_id,
            "season": int(main["season"].iloc[0]), "main_size": len(main),
            "trackman_size": len(history), "direct": direct,
            "aligned": matched, "coverage": coverage, "method": method,
            "accepted": accepted,
        })
    row_ids = np.concatenate(aligned_main).astype("U32")
    trackman_ids = np.concatenate(aligned_trackman).astype("U32")
    output = root / "outputs" / "trackman_pitch_alignment.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, row_id=row_ids, trackman_id=trackman_ids)
    detail = pd.DataFrame(details)
    detail.to_csv(root / "outputs" / "trackman_pitch_alignment_games.csv", index=False)
    report = {
        "team_maps": team_maps, "matched_games": len(detail),
        "accepted_games": int(detail["accepted"].sum()),
        "aligned_pitches": len(row_ids), "regular_rows": len(train),
        "row_coverage": float(len(row_ids) / len(train)),
        "coverage_quantiles": detail["coverage"].quantile(
            [0, .1, .25, .5, .75, .9, 1]
        ).to_dict(),
    }
    (root / "outputs" / "trackman_pitch_alignment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
