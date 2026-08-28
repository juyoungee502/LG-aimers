# Model guide

## Current candidate: v60

V60 keeps the confirmed v54 ensemble and adds a confidence correction for a
small, well-supported subset of F-regime rows. The official recent-game success
and middle rates are rounded fractions with a shared pitch-count denominator.
V60 reconstructs a conservative denominator from those fractions, then compares
six paired F-only CatBoost models: each pair is identical except that one member
sees the reconstructed recent-window confidence features. Only the averaged
paired difference is used; neither standalone model replaces v54.

The correction is applied only when the row is F-regime, the reconstructed
one-game denominator is at least 30, and current-season pitcher exposure is over
100 pitches. All other rows retain the v54 prediction.

Exact chronological validation on 2024 rows:

| Metric | v54 | v60 | Gain |
|---|---:|---:|---:|
| Overall BSS | 1023.917 | 1027.212 | +3.296 |
| R regime BSS | 1013.157 | 1013.157 | +0.000 |
| F regime BSS | 779.259 | 807.263 | +28.004 |
| First half | 1155.485 | 1159.959 | +4.473 |
| Second half | 880.449 | 882.567 | +2.118 |

All four quarter gains are positive: `+4.332`, `+4.615`, `+1.824`, and
`+2.412`. Returning players, roster changes, unchanged teams, player/team
changes, and high-exposure pitchers gain `+3.452`, `+2.983`, `+3.935`, `+2.524`,
and `+3.763`, respectively. Low-exposure rows are deliberately unchanged.

The six pairs were also split into two independent three-seed ensembles. At the
frozen production weight, their gains were `+22.550/+3.146` on 2023/2024 and
`+21.867/+2.958` on 2023/2024. Across both seed groups, the minimum quarter gain
was `+0.403`, the minimum affected roster gain was `+8.426`, and the minimum
pitcher-clustered bootstrap 5th percentile was `+0.823`.

These checks reduce historical roster and seed bias but cannot guarantee the
2025 public score. V54 scored **1113** on the public leaderboard and remains the
best confirmed result; v60 has not been submitted. The 1200-point target has not
yet been reached.

## GPU server: train, validate, test, and package

V54 artifacts and v59 audits are built automatically when missing. Run this single command
sequence on the server:

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v60.sh
```

The runner performs chronological/roster validation, GPU training, submission
packaging, and an isolated package smoke test. It creates:

- `submission_v60.zip`: code-submission ZIP
- `outputs/v60_oof_predictions.npz`: OOF diagnostics
- `training_v60.log`: complete build log
- `research/v59_group_stability.json`: independent-seed audit
- `outputs/results_v60.zip`: all deliverables bundled together

Copy the result bundle to the PC from PowerShell:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v60.zip" "outputs/results_v60.zip"
```

If v60 is selected for submission, extract and submit `submission_v60.zip` from
inside the result bundle.

## Design and data constraints

- Validation is chronological; a random split is not used for promotion.
- Added v54/v60 models exclude raw pitcher and batter IDs.
- The joint roster-robust component also excludes raw team IDs.
- V60 uses only official row-local recent-game rates and frozen training-history
  exposure; it does not aggregate or inspect other evaluation rows.
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
| v60 | 1027.212 | not submitted |

V26 demonstrates why a higher single-year local score is insufficient. It is
excluded from the current candidate path because its local gain did not transfer
to the public set. V55 additionally tested player/team-free, season-balanced
models and rejected them: their useful blend direction reversed between the
2023 and 2024 forward folds. V56 tested an eight-type historical pitch-selection
failure prior. Although its 2023/2024 yearly and quarterly gains were positive,
it is also excluded: its 2024 gain over v54 was only `+0.811`, the cohort absent
from the prior-season roster lost `-0.620`, and the pitcher-clustered bootstrap
5th percentile was `-0.705`.

V58 first tested reconstructed fraction confidence in a general direct model;
its effect was too small and seed-sensitive for promotion. V59 isolated the same
features inside F-only paired models. Two independent three-seed ensembles then
reproduced the same positive 2023/2024 direction, leading to the gated six-pair
v60 production correction.

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
