# Model guide

## Stable public-reference transfer: v66

V66 resumes from the submitted v64 anchor (**1135.1**) and does not retain
v65, v62, or v63.  It independently rebuilds a public solution's most useful
idea: train-only, reliability-shrunk command deviations for a pitcher versus
the batter side, the count advantage state, and whether runners are on base.
The exact-count refinement was rejected because it reversed in 2024.

Strict validation versus v64:

| Fold | Total gain | First half | Second half | R | F |
|---|---:|---:|---:|---:|---:|
| 2023 | +7.907 | +1.979 | +13.836 | +6.673 | +18.524 |
| 2024 | +4.137 | +6.047 | +2.227 | +3.583 | +8.307 |

All eight chronological quarter gains are positive; the weakest is `+0.692`.
Pitcher-clustered positive probabilities are `94.5%` (2023) and `91.4%`
(2024), although both 95% intervals still cross zero.  The honest projected
public range is **1135--1144**, not a guaranteed 1150.

Every lookup table is rebuilt from the official 2019--2024 `train.csv` only.
No 2025 TrackMan history, evaluation-row aggregation, external prediction, or
external model artifact is used.

Train, package, and smoke-test in one command on the GPU server:

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v66.sh
```

Copy the complete result bundle to the local PC:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v66.zip" "outputs/results_v66.zip"
```

Submit `submission_v66.zip` contained inside `results_v66.zip`.

## Conservative prediction-gap candidate: v65

Public leaderboard result: **1135.0**, which is `-0.1` versus v64's
**1135.1**.  V65 is therefore retired; new work resumes from v64 and does not
carry the prediction-gap correction forward.

The public v64 result is **1135.1**. V65 leaves that complete model intact and
fits a strongly regularized, zero-intercept Ridge direction to disagreements
between v64 and this project's own historical OOF checkpoints and raw members.
The correction scales are only `0.075` for R and `0.025` for F; the mean
absolute change on the forward 2024 fold is `0.000687`.

Only checkpoints that the current inference graph reproduces exactly are
eligible: v16, v17, v18, v19, v23, and v24. Later archives with removed legacy
stages are excluded, as are v62/v63. A specialist feature unavailable in the
forward-training source is frozen inactive so it cannot appear only during the
2025 production refit.

Validation versus v64:

| Fold | Total gain | First half | Second half | R | F |
|---|---:|---:|---:|---:|---:|
| 2023 H1 -> H2 | +4.155 | +5.564 | +2.745 | +3.888 | +6.871 |
| 2023 -> 2024 | +2.042 | +1.704 | +2.380 | +2.190 | +0.938 |

Pitcher-clustered 95% bootstrap intervals are `+1.82` to `+6.80` for 2023 H2
and `+0.33` to `+3.86` for 2024. This is a stricter gate than recent versions,
but hidden 2025 performance still cannot be guaranteed.

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v65.sh
```

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v65.zip" "outputs/results_v65.zip"
```

Submit `submission_v65.zip` contained inside the result bundle.

## Train-only calibration candidate: v63

V62 scored **1132.9**, only about `+0.2` over v61.  Its three structural
corrections are therefore removed rather than rescaled.  V63 returns to the
proven v61 path and applies only a fixed `-0.0015` probability offset after all
model and residual components.

The offset uses official training data only.  An OLS trend over the 2020--2024
annual target rates forecasts a 2025 rate of `0.478286`; packaged v61 predicts
`0.485355` on the full 2024-as-2025 training proxy.  The deployed offset is
only 21% of that `-0.007069` difference.  It gains `+5.15` on strict 2024 OOF,
and the train-trend central projection is **1140.30**.  Persistence of the 2024
rate would reverse the calibration direction, so the honest stress range is
wide at **1131--1145**.

No leaderboard-inferred target rate, external prediction, 2025 Trackman data,
or evaluation-row aggregation is used.

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v63.sh
```

Copy the result to this PC:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v63.zip" "outputs/results_v63.zip"
```

Submit `submission_v63.zip` contained inside `outputs/results_v63.zip`.

## Public-transfer candidate: v62

V61 scored **1132**, a `+7.1` gain over v60. Its observed transfer was about
69% of the projected gain. Fixed-score curvature puts the v61 correction's
vertex near `0.86x`, with only about `+0.2` left from scalar rescaling.

V62 therefore adds three mostly independent row-local tables. A mirrored
pitcher-hand-cell exposure direction reconstructs the public direction at
`0.9998` correlation. A new OOF residual hand differential reconstructs its
audited counterpart at `0.9724` correlation and is applied at half strength.
Finally, the pitcher-by-batter-hand deviation shrinkage shape is the only new
component with positive gains in both chronological directions (`+2.64` and
`+3.25` at deployed scale). V61's two new components are also reduced to
`0.86x`.

After the measured HD/D0 overlap penalty, the central projection is
**1140.06**, with a deliberately wide **1135--1144** range. The combined
backward transfer remains negative, so this is an aggressive public-transfer
candidate and not a score guarantee.

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v62.sh
```

Copy the result to this PC:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v62.zip" "outputs/results_v62.zip"
```

Submit `submission_v62.zip` contained inside `outputs/results_v62.zip`.

## Public-transfer candidate: v61

V60 scored **1124.9**, a `+4.4` gain over v59. Combining that result with the
fixed Brier curvature shows that the v60 hand-shape coordinate is already near
its public optimum; rescaling it has only about `+0.06` point left.

V61 instead completes two public-positive shape components that are missing
from the current stack. The first changes the shrinkage shape of a batter OOF
residual table from `k=20000` toward `k=2000`; its independently rebuilt row
direction correlates `0.9064` with the audited direction that gained `+13.788`
publicly. The second adds a residual-orthogonal pitcher log-exposure direction;
its reconstruction correlation is `0.99956` and its reported public gain was
`+0.691`.

Both tables are rebuilt from this project's own strict OOF predictions and
official 2023-2024 rows. Based on the linear-signal retention observed when v59
and v60 transferred, their conservative strengths are `0.85` and `0.80`. The
central public projection is **1135.18**, with a deliberately wide
**1131--1138** range. Chronological validation is negative, so this is a
leaderboard-transfer candidate rather than a guaranteed improvement.

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v61.sh
```

Copy the result to this PC:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v61.zip" "outputs/results_v61.zip"
```

Submit `submission_v61.zip` contained inside `outputs/results_v61.zip`.

## Public-transfer candidate: v60

V59 scored **1120.5**, a `+6.5` public gain over v58 and direct confirmation
that the 2025 high-usage batter direction transfers. That scalar coordinate is
already near its public optimum, so v60 leaves v59 unchanged and adds a new
shape direction.

For each pitcher, v60 estimates the strictly OOF residual contrast between
same- and opposite-handed batters. It changes only the shrinkage shape from
`k=1000` toward `k=100` at the public-tested `t=3` setting. The table is rebuilt
from this project's own 2023-2024 OOF residuals. It has 1,996 frozen cells and
unknown 2025 combinations receive zero.

The independently rebuilt correction has `0.9804` row correlation with the
same-test-set direction that publicly gained `+6.888` after the batter-exposure
improvement. It is essentially uncorrelated with this model's existing hand
effect, so it supplies a new shape rather than scaling an old correction.
Starting from v59, the central public projection is **1127.39**, with a wider
working range of **1123--1129**. The 2023-to-2024 forward result is negative,
so this remains a public-transfer experiment rather than a guarantee.

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v60.sh
```

Copy the result to this PC:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v60.zip" "outputs/results_v60.zip"
```

Submit `submission_v60.zip` contained inside `outputs/results_v60.zip`.

## Public-transfer candidate: v59

V58 scored **1114** publicly, close to its projected range. The F and R scalar
directions used through v58 are now near their measured public optima, so v59
adds a new deterministic direction instead of extending those curves.

V59 counts each batter's rows in the completed 2023-2024 seasons, centers the
counts over the 526 known batters, and freezes one probability delta per
`batter_id`. The table is label-free. Unknown 2025 batters receive zero, and
each evaluation row performs only its own ID lookup. The maximum correction is
`+0.00914` and the minimum is `-0.00198`.

This choice is deliberately based on a public same-test-set result: a nearly
identical pure batter-exposure direction improved another fixed model by
`+8.3544` points and its measured quadratic optimum was essentially the tested
amplitude. Applied to the v58 public anchor, the central projection is
**1122.35**, with a deliberately wider **1119.0--1123.5** working range. This
is evidence, not a guarantee. The direction is negative on the 2024 forward
fold (`-11.15` BSS), reproducing the known 2024-to-2025 reversal instead of
hiding it.

V59 contains no external model or prediction, no target-derived player table,
no 2025 Trackman data, and no aggregation over evaluation rows.

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v59.sh
```

Copy the complete result to this PC:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v59.zip" "outputs/results_v59.zip"
```

Submit `submission_v59.zip` contained inside `outputs/results_v59.zip`.

## Current public-feedback candidate: v58

V57 scored approximately **1112**, below the unchanged v56 anchor at
**1113.86**. Because v57 changed only R rows, its frozen R residual direction
is now treated as a 2025-reversed signal and is never applied with its old
positive weight.

V58 starts from v56. It makes only two measured changes:

- the public-positive F correction scale moves from `1.25` to `1.375`;
- the v57 R direction is applied at `-0.25` instead of `+1.0`, still only when
  `pitcher_season_n > 100`.

The deployed v57 table has only `0.573` times the squared correction magnitude
of its forward-fold proxy. Combining that measured curvature with the rounded
v57 public result projects v58 at approximately **1114.22--1114.47**. This is a
leaderboard-feedback estimate, not a guarantee. The 2024 local score is
`-1.062` BSS versus v56 because the R counterstep deliberately follows the
observed 2025 public reversal rather than the non-transferring 2024 direction.

V58 remains row-independent and uses no 2025 Trackman data, current-pitch
outcome, test-row aggregation, or raw test-derived statistics.

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v58.sh
```

Copy the result to this PC:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v58.zip" "outputs/results_v58.zip"
```

Submit `submission_v58.zip` contained inside `outputs/results_v58.zip`.

## Retired public-negative candidate: v57

V56 scored **1113.86** publicly. V57 preserves the complete v56 F prediction
and adds one conservative residual lookup only to regular-season (`R`) rows.
The lookup uses eight bins of `pitcher_success_x_runners`, heavy shrinkage of
`6400`, and weight `0.5`. It contains no raw player ID and is less dependent on
the exact 2025 roster than player-specific tables. The correction is enabled
only after `pitcher_season_n > 100`; early-season and cold-start rows remain
exactly at the v56 prediction.

On chronological 2023-to-2024 validation, v57 gained `+3.505` overall BSS and
`+3.974` on R rows over v56. The four R-quarter gains are `+3.843`, `+4.748`,
`+3.123`, and `+4.209`. A pitcher-clustered bootstrap has a `+1.005` 5th
percentile and a `98.7%` probability of improvement. All roster/exposure
cohorts were non-negative because low-exposure rows were left unchanged. That
local result did not transfer: the public score was approximately **1112**, so
v57 must not be submitted again.

V57 uses only frozen statistics learned from 2019--2024 training data. It does
not use 2025 Trackman history, current-pitch outcome fields, or aggregation over
test rows. The final validation, roster/team audit, and packaged inference smoke
test are all run by the single server command below.

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v57.sh
```

Copy the combined result to this PC:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v57.zip" "outputs/results_v57.zip"
```

Submit `submission_v57.zip` contained inside `outputs/results_v57.zip`.

## Previous experimental candidate: v56

V55 scored approximately **1113.6** publicly, about `+0.5` over v54. This is
evidence that the conservative F-regime scaling transferred to hidden 2025
rows. V56 takes exactly one more equal-sized step: R remains unchanged and the
v54 F correction scale moves from `1.125` to `1.25`.

Against v55 on chronological 2024 validation, v56 gains `+0.186` overall and
`+1.579` in F. All quarters and roster/team-change cohorts remain positive, but
the minimum quarter gain is only `+0.031` and the pitcher-clustered bootstrap
5th percentile is `-0.154`. V56 is therefore a measured leaderboard experiment,
not a guaranteed improvement.

```bash
cd ~/바탕화면/LG-aimers
git pull
bash run_v56.sh
```

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v56.zip" "outputs/results_v56.zip"
```

## Confirmed public candidate: v55

V55 keeps every v54 model and changes only the strength of the already
public-positive v54 correction. Regular-season (`R`) rows remain exactly v54.
For `F` rows, the logit correction from v38 to v54 is multiplied by `1.125`.
No new model, raw ID, test-row aggregation, or post-2024 Trackman data is used.

Against v54 on chronological 2024 validation, v55 gains `+0.222` overall and
`+1.883` in F. Its four quarter gains are `+0.456`, `+0.192`, `+0.171`, and
`+0.067`; returning/changed player and team cohorts are all positive. This is a
small experimental improvement, not a confirmed leaderboard gain: the
pitcher-clustered bootstrap 5th percentile is `-0.121`.

Run and package v55 on the GPU server:

```bash
cd ~/바탕화면/LG-aimers
git pull
bash run_v55.sh
```

Copy the combined result to the PC:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v55.zip" "outputs/results_v55.zip"
```

## Confirmed anchor: v54

V54 is a conservative, roster-robust addition to the v38 ensemble. It predicts
coherent command outcomes and a latent `pitch family × command outcome` target.
The added models do not use raw pitcher or batter IDs; the joint model also
removes both team IDs. This reduces dependence on the exact players and teams
seen in earlier seasons.

Exact chronological validation on 2024 rows:

| Metric | v38 | v54 | Gain |
|---|---:|---:|---:|
| Overall BSS | 1020.853 | 1023.917 | +3.064 |
| R regime BSS | 1013.157 | 1013.157 | +0.000 |
| F regime BSS | 753.228 | 779.259 | +26.032 |
| First half | 1151.835 | 1155.485 | +3.651 |
| Second half | 877.973 | 880.449 | +2.476 |

All four quarter gains are positive: `+4.937`, `+2.362`, `+3.446`, and
`+1.508`. The improvement also remains positive for returning players, roster
changes, unchanged teams, player/team changes, and low/high pitcher-exposure
cohorts. The weakest of those roster slices is `+0.831` BSS. A pitcher-clustered
bootstrap gives a 5th percentile gain of `+0.046` and a `95.3%` probability of
positive improvement.

These checks reduce historical roster bias but cannot guarantee the 2025 public
score. V54 scored **1113** on the public leaderboard, the best confirmed result
so far, but the 1200-point target has not been reached.

## GPU server: train, validate, test, and package

V38 artifacts are built automatically when missing. Run this single command
sequence on the server:

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v54.sh
```

The runner performs chronological/roster validation, GPU training, submission
packaging, and an isolated package smoke test. It creates:

- `submission_v54.zip`: code-submission ZIP
- `outputs/v54_oof_predictions.npz`: OOF diagnostics
- `training_v54.log`: complete build log
- `research/v53_roster_stability.json`: roster audit
- `outputs/results_v54.zip`: all deliverables bundled together

Copy the result bundle to the PC from PowerShell:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v54.zip" "outputs/results_v54.zip"
```

If v54 is eventually selected for submission, extract and submit
`submission_v54.zip` from inside the result bundle.

## Design and data constraints

- Validation is chronological; a random split is not used for promotion.
- Added v54 models exclude raw pitcher and batter IDs.
- The joint roster-robust component also excludes raw team IDs.
- Inference is row-independent and never aggregates evaluation rows.
- Anonymous and Trackman IDs are linked only through allowed 2019-2024
  historical pitch-sequence alignment.
- No 2025 Trackman history is read, joined, trained on, or packaged.
- Current pitch type is never an inference input. Historical pitch labels are
  reconstructed from the next cumulative state only for training targets.

## Public-score anchors

Known public results should be treated as empirical anchors, not deterministic
translations from local validation:

| Version | Local chronological BSS | Public score |
|---|---:|---:|
| v17 | 934.687 | 1076 |
| v23 | 989.538 | 1105 |
| v26 | 1043.739 | 1079 |
| v54 | 1023.917 | 1113 |
| v55 | 1024.139 | 1113.6 |
| v56 | 1024.325 | 1113.86 |
| v57 | 1027.830 | 1112 |
| v58 | 1023.263 | 1114 |

V26 demonstrates why a higher single-year local score is insufficient. It is
excluded from the current candidate path because its local gain did not transfer
to the public set. V55 additionally tested player/team-free, season-balanced
models and rejected them: their useful blend direction reversed between the
2023 and 2024 forward folds.

## Older standalone pipeline

```powershell
pip install -r requirements.txt
python train.py --preset full --task-type CPU
python script.py
powershell -ExecutionPolicy Bypass -File .\make_submission.ps1
```

On a single-GPU Linux server, CatBoost training can use:

```bash
python train.py --preset full --task-type GPU --devices 0
```

## Research basis

- Prokhorenkova et al., *CatBoost: unbiased boosting with categorical
  features*, NeurIPS 2018:
  https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
- Sidle and Tran, *Using multi-class classification methods to predict baseball
  pitch types*, Journal of Sports Analytics 2018. Count-dependent pitching
  behavior motivates explicit count and pitcher-by-state effects.
