# Model guide

## Current candidate: v54

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

V26 demonstrates why a higher single-year local score is insufficient. It is
excluded from the current candidate path because its local gain did not transfer
to the public set. V55 additionally tested player/team-free, season-balanced
models and rejected them: their useful blend direction reversed between the
2023 and 2024 forward folds. V56 tested an eight-type historical pitch-selection
failure prior. Although its 2023/2024 yearly and quarterly gains were positive,
it is also excluded: its 2024 gain over v54 was only `+0.811`, the cohort absent
from the prior-season roster lost `-0.620`, and the pitcher-clustered bootstrap
5th percentile was `-0.705`.

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
