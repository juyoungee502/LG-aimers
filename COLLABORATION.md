# 협업 및 실험 관리 규칙

## 브랜치

`main`에는 검증된 코드만 둔다. 모든 실험은 아래 형식의 브랜치에서 진행한다.

```text
experiment/<이름>-<모델 또는 목적>
fix/<이름>-<수정 내용>
```

예: `experiment/junseo-catboost`, `experiment/juyoung-lgb-ensemble`

작업 시작 전에는 원격 상태를 반영하고 `main`에서 새 브랜치를 만든다.

```powershell
git switch main
git pull --ff-only origin main
git switch -c experiment/<이름>-<실험명>
```

`--ff-only`는 의도하지 않은 merge 커밋이 생기는 것을 방지한다. 이미 작업 중인
브랜치에서는 `git fetch origin` 후 필요한 경우 `git rebase origin/main`을 사용한다.

## 실험 구분

각 모델은 `experiments/<experiment_id>/manifest.json`으로 식별한다. 새 실험을 만들
때는 `experiments/_template/manifest.json`을 복사하고 아래 항목을 반드시 기록한다.

- 담당자와 브랜치
- 학습 및 추론 진입점
- 모델과 피처 버전
- 학습 시즌과 검증 시즌
- seed와 주요 하이퍼파라미터
- 검증 지표 및 대회 제출 점수
- 재현 명령

실험 ID는 `expNNN_<모델>_<핵심변경>` 형식을 권장한다.

## Git에 포함하는 파일

포함: 소스 코드, 설정, manifest, 작은 검증 요약 JSON/CSV.

제외: 원본 데이터, 학습 모델, 전체 예측 배열, 제출 ZIP, 로그. `.gitignore`에서
`data/`, `model/`, `submit/model/`, `outputs/`, `submissions/`를 제외한다. 이미 Git이
추적 중인 산출물은 별도 합의 없이 삭제하거나 추적 해제하지 않는다.

## Pull Request 확인사항

1. 시간 순서 검증인지 확인한다.
2. 테스트 행 간 집계와 사후 정보 사용이 없는지 확인한다.
3. 학습과 추론에서 동일한 피처 코드가 사용되는지 확인한다.
4. manifest에 실행 명령과 지표를 기록한다.
5. 기존 최종 제출 경로를 덮어쓰지 않는지 확인한다.

커밋 메시지는 `feat(catboost): ...`, `exp(lgb): ...`, `fix(inference): ...`처럼
변경 목적과 모델을 드러내도록 작성한다.
