"""Reliably align official main-table pitches to historical Trackman logs.

The two tables have different anonymous identifiers.  Games are linked by
season/date fingerprint, teams, pitch counts, and pre-pitch state sequences;
only games with at least 90% sequence coverage are retained.
"""
from __future__ import annotations

import json
from collections import Counter
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


def edge_tensor(main, history, main_teams, history_teams):
    """Schedule overlap for every anonymous-to-named directed team edge."""
    size = len(main_teams)
    tensor = np.zeros((size, size, size, size), dtype=np.float32)
    main_edges, history_edges = {}, {}
    for i, home in enumerate(main_teams):
        for j, away in enumerate(main_teams):
            if i != j:
                rows = main.loc[main["home"].eq(home) & main["away"].eq(away)]
                main_edges[i, j] = Counter(zip(rows["month"], rows["weekday"]))
    for i, home in enumerate(history_teams):
        for j, away in enumerate(history_teams):
            if i != j:
                rows = history.loc[
                    history["home"].eq(home) & history["away"].eq(away)
                ]
                history_edges[i, j] = Counter(zip(rows["month"], rows["weekday"]))
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            for u in range(size):
                for v in range(size):
                    if u != v:
                        tensor[i, j, u, v] = sum(
                            (main_edges[i, j] & history_edges[u, v]).values()
                        )
    return tensor


def mapping_score(tensor, permutation):
    return float(sum(
        tensor[i, j, permutation[i], permutation[j]]
        for i in range(len(permutation))
        for j in range(len(permutation)) if i != j
    ))


def climb_mapping(tensor, start):
    permutation = np.asarray(start, dtype=np.int16).copy()
    best = mapping_score(tensor, permutation)
    while True:
        choice = None
        for i in range(len(permutation)):
            for j in range(i + 1, len(permutation)):
                candidate = permutation.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                value = mapping_score(tensor, candidate)
                if value > best + 1e-9:
                    best, choice = value, candidate
        if choice is None:
            return best, permutation
        permutation = choice


def infer_team_maps(main_games, history_games, restarts=600):
    """Resolve team IDs from the season schedule graph with random restarts."""
    rng = np.random.default_rng(20260814)
    maps, diagnostics = {}, []
    for season in sorted(main_games["season"].unique()):
        main = main_games.loc[main_games["season"].eq(season)]
        history = history_games.loc[history_games["season"].eq(season)]
        main_teams = sorted(set(main["home"]) | set(main["away"]))
        history_teams = sorted(set(history["home"]) | set(history["away"]))
        if len(main_teams) != 10 or len(history_teams) != 10:
            raise ValueError((season, main_teams, history_teams))
        tensor = edge_tensor(main, history, main_teams, history_teams)
        starts = [np.arange(10, dtype=np.int16)]
        starts.extend(rng.permutation(10) for _ in range(restarts))
        solutions = {}
        for start in starts:
            score, permutation = climb_mapping(tensor, start)
            solutions[tuple(map(int, permutation))] = score
        ranked = sorted(solutions.items(), key=lambda item: item[1], reverse=True)
        permutation, score = ranked[0]
        mapping = {
            int(main_teams[i]): str(history_teams[permutation[i]])
            for i in range(10)
        }
        maps[int(season)] = mapping
        diagnostics.append({
            "season": int(season), "score": score,
            "second": ranked[1][1], "gap": score - ranked[1][1],
            "mapping": mapping,
        })
        print(json.dumps(diagnostics[-1]), flush=True)
    return maps, diagnostics


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
    team_maps, team_diagnostics = infer_team_maps(main_games, history_games)
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
        "team_maps": team_maps, "team_diagnostics": team_diagnostics,
        "matched_games": len(detail),
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
