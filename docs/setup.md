# 실행 방법

저장소 루트에서 사용할 Python 환경을 활성화한 뒤 실행합니다.
기존 Windows 환경은 `baseline/Scripts/python.exe`를 `python` 대신 사용할 수 있습니다.
필요한 라이브러리는 사용할 노트북의 설치 셀을 참고하세요.

```bash
# 기존 단위 테스트 (GPU·모델 불필요)
python -m unittest discover -s tests -v

# 필요한 모델·데이터·실험 A 산출물 준비 후 실행
python scripts/run_nb.py notebooks/experiments/AI_Challenge2_expB_vlm_only.ipynb

# 결과 비교
python scripts/compare_experiments.py

# B와 C가 갈린 문항에서 OCR이 실제로 기여했는지 확인 (--list로 문항별 출력)
python scripts/analyze_discordant.py
```

### 제출 파일 생성

E1 노트북 뒤쪽 **제출 추론** 섹션이 `data/test.csv` 6,714문항을 검증과 동일한 설정으로 추론해
`output/exp_E1_test/` 아래에 제출 파일을 만듭니다. 셀 1 → 3 → 5 → 7을 실행한 뒤 해당 섹션으로 가세요.
파일 이름은 `submission_<모델>_<방식>_v<버전>.csv` 형식입니다(예: `submission_qwen35-9b_zeroshot_v1.csv`).
방식은 `SUBMISSION_METHOD`로 정하고, 버전은 비워 두면 같은 모델·방식의 기존 파일을 보고 다음 번호가 붙어
회차끼리 덮어쓰지 않습니다. 이름을 통째로 고정하려면 `SUBMISSION_NAME`을 쓰세요.
RTX 5060 Ti 기준 약 90분이고, 문항마다 `predictions.jsonl`에 기록하므로 중단 후 이어서 돌릴 수 있습니다.

`SUBMISSION_CSV` 한 줄만 바꾸면 대상이 바뀌고 출력 폴더도 `output/exp_E1_<대상이름>`으로 따라갑니다.
정답이 있는 CSV(`data/train.csv` 등)를 주면 채점 모드로 동작하며 제출 파일은 만들지 않습니다.
기존 실험 폴더를 덮어쓰려 하면 `run_meta.json` 대조에서 멈춥니다.
세부 설정은 노트북 마지막의 **사용 가이드**에 정리했습니다.

파서가 실패한 문항은 a~d 로짓 비교로 채웁니다. 특정 글자를 기본값으로 두면
`output/_legacy/submission.csv`처럼 예측이 한쪽으로 쏠리므로(`a` 42%) 그 방식은 쓰지 마세요.

Colab에서 돌리려면 [E1 Colab 노트북](../notebooks/experiments/AI_Challenge2_expE1_qwen35_9b_colab.ipynb)을
쓰세요. **A100 GPU + bf16**을 권장하며, VRAM을 감지해 L4·T4에서는 4bit로 자동 전환합니다.
데이터는 Drive의 `ssafy-16-2-ai.zip`, 공통 코드는 저장소 clone, 결과는 Drive에 사본을 남겨 재개할 수 있습니다.

CoT 제출 추론은 [E2 Colab 노트북](../notebooks/experiments/AI_Challenge2_expE2_qwen35_9b_cot_colab.ipynb)입니다.
E1 Colab에서 `scripts/build_e2_colab_notebook.py`로 생성하므로 직접 고치지 말고 스크립트를 고친 뒤 다시 생성하세요.
E1과 다른 부분은 프롬프트(`SYSTEM_COT`, `cot=True`), 출력 상한(512토큰), 파서(`parse_choice_cot`),
폴백(비-CoT 재질의 → 그래도 실패하면 로짓), 출력 폴더(`exp_E2_<대상>`)뿐입니다.
CoT 입력에서는 다음 토큰이 선지 글자가 아니므로 **E1의 로짓 폴백을 그대로 쓰면 안 됩니다.**
512토큰 안에 결론을 못 낸 출력은 파싱에 성공해도 폴백으로 넘깁니다(`RETRY_ON_TRUNCATED`).

기본 대상은 E1과 같은 **`test.csv`**입니다. 학습이 없으므로 train.csv를 꼭 거칠 필요가 없습니다.
제로샷이라 train.csv는 "학습에 쓴 데이터"가 아니라 그냥 정답이 붙은 문제지이고, 용도는
*제출하지 않고 미리 재보는 것* 하나뿐입니다. 이미 `output/exp_E1_test/`에 같은 A100 bf16으로
돌린 E1 zeroshot 6,714문항 예측과 Kaggle 0.94727이 있으므로, E2를 test.csv로 돌려 제출하면
같은 규모에서 바로 비교됩니다(E2 약 4~5시간, E1 재실행 불필요).

16번 섹션이 대상에 따라 두 가지로 동작합니다. **정답이 없을 때**(test.csv)는 두 예측이 갈린
문항 수와 분포를 보고, 제출 점수가 나오면 `kaggle_delta(...)`로 보수적인 McNemar 판정을 합니다
(점수 차이가 `c − b`를 주고 불일치 문항 수가 `b + c`의 상한이라, 불일치 전부가 정오를 가른다고
놓으면 가장 불리한 p값이 나옵니다). **정답이 있을 때**(train.csv)는 문항 단위 McNemar 검정과
정확도 차 신뢰구간을 계산하고, 672 분할 ID를 Drive에 올려 두면 그 부분집합 점수도 함께 찍습니다.

제출 횟수가 빠듯하면 `SUBMISSION_CSV = DATA_ROOT / "train.csv"`로 바꾸고 E1도 같은 CSV로
돌리세요(약 37분). 그 수치는 **A~D 비교표에 넣지 마세요.** 그 표는 672 분할 기준입니다.

로컬 노트북은 저장소 루트 또는 `notebooks/` 하위 폴더에서 열고 첫 코드 셀부터 실행하세요.
경로 설정 셀이 저장소 루트를 찾아 `src/`를 import 경로에 추가합니다.
Colab 노트북은 Colab에서 열고 자체 설치·Drive 설정 셀을 실행합니다.
`run_nb.py`는 `!pip`, `%pip` 등 IPython 전용 명령을 지원하지 않습니다.

### E4 · E5 실행

E4는 Qwen3.5-27B 가중치(약 52 GiB)가 필요합니다. 끊기면 같은 명령을 다시 실행하면 이어 받습니다.

```bash
baseline/Scripts/python.exe scripts/download_challenge_models.py Qwen/Qwen3.5-27B
baseline/Scripts/python.exe scripts/run_nb.py notebooks/experiments/AI_Challenge2_expE4_qwen35_27b.ipynb
baseline/Scripts/python.exe scripts/run_nb.py notebooks/experiments/AI_Challenge2_expE5_qlora_9b.ipynb
```

E4 검증을 끝낸 뒤 test 제출만 돌릴 때는 설정 셀과 제출 셀만 실행합니다(약 3.8시간, 끊기면 같은 명령으로 이어감).

```bash
baseline/Scripts/python.exe scripts/run_nb.py notebooks/experiments/AI_Challenge2_expE4_qwen35_27b.ipynb --from "제출 추론"
```

Colab A100 80GB가 있으면 `AI_Challenge2_expE4_qwen35_27b_vllm_colab.ipynb`로 bf16 제출을 약 1시간에 돌릴 수 있습니다.
val 672를 먼저 돌려 로컬 E4와 대조한 뒤 test로 넘어갑니다.

GPU가 하나이므로 E4와 E5는 순서대로 돌리세요. 다운로드는 GPU를 쓰지 않으므로 E5 학습과 동시에 받아도 됩니다.
두 노트북 모두 문항·스텝마다 기록하고 `RESUME = True`면 끊긴 지점부터 이어갑니다.
E4에서 OOM이 나면 `CPU_LAST_LAYERS`를 2, 4로 올립니다(느려집니다).
노트북은 `scripts/build_e4_notebook.py`, `build_e5_notebook.py`로 생성하므로 직접 고치지 말고 스크립트를 고친 뒤 다시 생성하세요.

## 로컬 전용 폴더 (Git 제외)

| 폴더 | 내용 |
| --- | --- |
| `baseline/` | 기존 Python 가상환경 |
| `data/` | 대회 데이터 |
| `downloads/` | 모델·설치 파일 |
| `output/` | 실험 결과·OCR 캐시·로그 |

데이터·모델·결과는 기존 루트 경로를 사용합니다.
이전 업로드 시 실행 출력을 정리한 노트북 원본은
`output/_legacy/git_upload_notebook_backups/`에 보관되어 있습니다.
2026-09-22에 초기 베이스라인 산출물(`qwen2_5_vl_3b_lora/`, `submission.csv`, 노트북 백업)을
`output/_legacy/`로, 캐시 복구 임시 파일을 `output/ocr_cache/_archive/`로 옮겼습니다.
