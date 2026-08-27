"""Audited policy constants for the v26 Pareto temporal portfolio.

The policy was selected on chronological 2023->2024 transfers while requiring
positive gains on every audited half, quarter, and sufficiently large month.
It is kept separate from v25 so that the conservative v25 submission remains
fully reproducible.
"""

REGULAR_POLICY = (
    {"type": "one_d", "kind": "numeric", "column": "asof_pitcher_success_rate", "bins": 16, "shrink": 6400., "scale": .5, "weight": .9874766911068117},
    {"type": "one_d", "kind": "numeric", "column": "asof_pitcher_prev5_game_success_rate", "bins": 16, "shrink": 400., "scale": .25, "weight": .21733195597341798},
    {"type": "one_d", "kind": "numeric", "column": "batter_middle_season_s25", "bins": 4, "shrink": 1600., "scale": 1., "weight": .13020844756212765},
    {"type": "one_d", "kind": "numeric", "column": "pitcher_middle_trend_1_3", "bins": 8, "shrink": 6400., "scale": .5, "weight": .7178049785463033},
    {"type": "one_d", "kind": "numeric", "column": "batter_team_id", "bins": 16, "shrink": 6400., "scale": .25, "weight": .24066926592909646},
    {"type": "one_d", "kind": "numeric", "column": "asof_pitcher_prev3_game_middle_rate", "bins": 8, "shrink": 6400., "scale": .5, "weight": .3785011298615493},
    {"type": "pair", "column": "pitcher_success_x_runners", "context": "num_runners_on", "bins": 4, "shrink": 400., "scale": .5, "weight": .31270557457798454},
    {"type": "pair", "column": "pitcher_middle_season_rate", "context": "balls_before", "bins": 4, "shrink": 400., "scale": .5, "weight": .1878631350999785},
    {"type": "pair", "column": "pitcher_season_minus_prior", "context": "inning_bucket", "bins": 8, "shrink": 1600., "scale": .25, "weight": .3950818965480313},
    {"type": "pair", "column": "pitcher_success_x_runners", "context": "pitcher_hand", "bins": 8, "shrink": 400., "scale": .5, "weight": .25615842417015944},
    {"type": "pair", "column": "pitcher_success_x_runners", "context": "pressure_state", "bins": 8, "shrink": 400., "scale": .5, "weight": .23342565352096153},
    {"type": "pair", "column": "pitcher_middle_season_s100", "context": "batter_hand", "bins": 8, "shrink": 400., "scale": .25, "weight": .3227371558149593},
)

FUTURES_POLICY = (
    {"type": "one_d", "kind": "numeric", "column": "pitcher_season_success_count", "bins": 16, "shrink": 25., "scale": .5, "weight": .6},
    {"type": "one_d", "kind": "numeric", "column": "asof_pitcher_reverse_rate", "bins": 8, "shrink": 25., "scale": .5, "weight": .4},
    {"type": "one_d", "kind": "numeric", "column": "pitcher_strike_x_same_hand", "bins": 16, "shrink": 25., "scale": .25, "weight": .35},
    {"type": "one_d", "kind": "numeric", "column": "two_strike", "bins": 4, "shrink": 400., "scale": .5, "weight": .1},
    {"type": "one_d", "kind": "numeric", "column": "pitcher_reverse_delta_x_2strike", "bins": 8, "shrink": 400., "scale": .25, "weight": 1.},
    {"type": "one_d", "kind": "numeric", "column": "pitcher_reverse_x_2strike", "bins": 8, "shrink": 400., "scale": .5, "weight": .15},
    {"type": "pair", "column": "pitcher_season_success_s50", "context": "batter_hand", "bins": 4, "shrink": 400., "scale": .5, "weight": .45},
    {"type": "pair", "column": "pitcher_season_success_s100", "context": "batter_hand", "bins": 4, "shrink": 1600., "scale": 1., "weight": .35},
    {"type": "pair", "column": "pitcher_reverse_season_s25", "context": "leverage_bucket", "bins": 4, "shrink": 100., "scale": .5, "weight": .5},
    {"type": "pair", "column": "pitcher_reverse_x_advantage", "context": "pitcher_hand", "bins": 8, "shrink": 100., "scale": .5, "weight": .65},
)

FUTURES_CALIBRATION_POLICY = (
    {"type": "probability_pair", "context": "inning_bucket", "bins": 8,
     "shrink": 100., "scale": .1, "weight": .25},
)
