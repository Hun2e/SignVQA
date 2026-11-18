"""Update the local OCR notebook without executing OCR or touching its cache."""

from _project import ROOT, prepare_notebook
import ast
import json
from pathlib import Path

path = ROOT / 'notebooks/ocr/AI_Challenge2_OCR_cache.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))

def set_cell(index, text):
    nb['cells'][index]['source'] = text.strip().splitlines(keepends=True)

set_cell(0, '''# 로컬 OCR — 저장하고 다음 날 이어서 실행

OCR 결과 한 장이 체크포인트 한 개입니다. **한 장마다 JSONL에 기록하고 디스크 동기화**합니다.
다음 날 같은 노트북에서 **1 → 2 → 3 → 5 → 6**을 실행하면 저장된 ID를 건너뜁니다.
선택 보정 셀 4는 실행하지 않아도 됩니다. 마지막에 셀 7로 결과를 확인하세요.

- `RUN_HOURS = 2.0`: 오늘은 약 2시간 동안 실행하고 현재 이미지 완료 후 자동 중지.
- `RUN_HOURS = None`: 시간 제한 없이 실행. `MAX_IMAGES_PER_RUN`으로 장수 제한도 가능.
- 즉시 멈추려면 Jupyter/VS Code의 **실행 중지(Interrupt)**를 누르고 저장 메시지를 확인한 뒤 종료.
- 강제 종료/전원 차단 시 처리 중인 이미지나 마지막 미완성 기록은 다음 날 다시 처리할 수 있습니다.
- 절전 중에는 OCR이 진행되지 않습니다. 자동 중지는 컴퓨터 종료 기능이 아닙니다.
- 같은 결과 파일에 두 노트북/프로세스를 동시에 실행하지 마세요.

결과: `output/ocr_cache/train_spotting.jsonl`
진행 요약: `output/ocr_cache/train_spotting.progress.json`
완료 여부는 요약이 아니라 JSONL에서 다시 읽습니다. 기존 로컬 결과도 이어서 사용합니다.
''')
set_cell(2, '''import gc, json, os, sys, time
from pathlib import Path

# Find the project even when the notebook is opened from a nested folder.
PROJECT_DIR = next((p for p in [Path.cwd(), *Path.cwd().parents]
                    if (p / 'src' / 'ocr_engine.py').is_file() and (p / 'data').is_dir()), None)
assert PROJECT_DIR is not None, '프로젝트 폴더에서 노트북을 실행하세요.'
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'

import pandas as pd
import torch
from tqdm.auto import tqdm
import ocr_engine as oe
from ocr_checkpoint import durable_append, atomic_json, resume_ids, validate_metadata

print('프로젝트:', PROJECT_DIR)
print('torch', torch.__version__, '/ CUDA', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
''')
set_cell(3, '''## 2. 파라미터

`RUN_HOURS` 또는 `MAX_IMAGES_PER_RUN`으로 오늘 실행할 분량을 정하세요.
둘 다 설정하면 먼저 도달한 조건으로 멈춥니다. 시간은 모델 로딩 이후부터 계산하며,
이미지 한 장 처리가 끝난 시점에 검사하므로 마지막 이미지 처리 시간만큼 초과할 수 있습니다.
기존 결과를 이어 쓸 때는 `SPLIT`, `PROMPT`, `MAX_NEW_TOKENS`, 모델/픽셀 설정을 유지하세요.
`LIMIT`은 데이터 앞 N장만 대상으로 삼는 시험용 옵션이고, 오늘 실행할 장수는 `MAX_IMAGES_PER_RUN`입니다.
''')
set_cell(4, '''SPLIT = 'train'             # train / test / dev / val672
MAX_NEW_TOKENS = 1024
PROMPT = 'Spotting:'
LIMIT = None

RUN_HOURS = 2.0             # 오늘 약 2시간 실행. 제한 없으면 None
MAX_IMAGES_PER_RUN = None  # 오늘 100장만 처리하려면 100
PROGRESS_EVERY = 25        # 진행 요약 파일 갱신 주기. OCR 결과는 매 장 저장
assert RUN_HOURS is None or RUN_HOURS > 0
assert MAX_IMAGES_PER_RUN is None or (isinstance(MAX_IMAGES_PER_RUN, int) and MAX_IMAGES_PER_RUN > 0)
assert isinstance(PROGRESS_EVERY, int) and PROGRESS_EVERY > 0

DATA_DIR = Path('data')
CACHE_DIR = Path('output/ocr_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = CACHE_DIR / f'{SPLIT}_spotting.jsonl'
RESUME_PATH = CACHE_DIR / f'{SPLIT}_spotting_local_resume.jsonl'
if RESUME_PATH.is_file():
    OUT_PATH = RESUME_PATH
PROGRESS_PATH = OUT_PATH.with_suffix('.progress.json')
oe.SPOTTING_PROMPT = PROMPT
print(f'{SPLIT} / {PROMPT!r} / max_new_tokens={MAX_NEW_TOKENS}')
print('출력:', OUT_PATH.resolve())
print('실행 시간 제한:', RUN_HOURS, '시간 / 장수 제한:', MAX_IMAGES_PER_RUN)
''')
set_cell(9, '''## 5. 이어하기 준비

JSONL에 저장된 이미지 ID를 읽습니다. 프롬프트·토큰 한도·메타데이터가 다르면 중단합니다.
강제 종료로 마지막 줄이 미완성이면 원본을 `.bak`으로 보관하고 그 줄만 복구합니다.
같은 OCR 결과의 중복 ID는 백업 후 하나만 남깁니다. 처리 시간 차이는 무시합니다.
파일 중간 손상이나 OCR 결과가 서로 다른 중복 ID는 오류를 표시합니다.
''')
set_cell(10, '''import importlib
import ocr_checkpoint
importlib.reload(ocr_checkpoint)
from ocr_checkpoint import durable_append, atomic_json, resume_ids, validate_metadata

def prepare_resume():
    settings = {
        'split': SPLIT, 'rows': len(df), 'max_new_tokens': MAX_NEW_TOKENS,
        'prompt': PROMPT, 'model': str(oe.OCR_MODEL_DIR),
        'max_pixels': oe.OCR_MAX_PIXELS,
        'upscale_threshold': oe.SPOTTING_UPSCALE_THRESHOLD,
    }
    validate_metadata(OUT_PATH.with_suffix('.meta.json'), settings)
    done = resume_ids(OUT_PATH, MAX_NEW_TOKENS, PROMPT)
    atomic_json(OUT_PATH.with_suffix('.meta.json'), settings)
    return done, df[~df.id.isin(done)].copy()

done, todo = prepare_resume()
print(f'대상 {len(df)}장 / 완료 {len(df) - len(todo)}장 / 남음 {len(todo)}장')
''')
set_cell(11, '''## 6. OCR 실행 — 매 이미지 저장 / 시간 제한 후 중지

이 셀만 재실행해도 완료 ID를 다시 읽어 중복 처리를 방지합니다.
중지 버튼으로 중단하면 처리 중인 이미지는 다음 실행에서 다시 처리합니다.
`저장 완료` 메시지를 확인한 뒤 프로그램을 닫으세요.
''')
set_cell(12, '''# Re-read the disk checkpoint even if only this cell is rerun.
done, todo = prepare_resume()
if todo.empty:
    print('이미 모두 완료되었습니다.')
else:
    assert torch.cuda.is_available(), '로컬 CUDA GPU를 사용할 수 있는 환경에서 실행하세요.'
    oe.SPOTTING_PROMPT = PROMPT
    model, proc = oe.load_ocr()
    n_new, elapsed_total, truncated_count = 0, 0.0, 0
    started = time.perf_counter()
    status = 'running'
    target_ids = set(df.id)

    def save_progress(state):
        completed = len(target_ids & done)
        atomic_json(PROGRESS_PATH, {
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': state, 'total': len(df), 'completed': completed,
            'remaining': len(df) - completed, 'new_this_run': n_new,
            'elapsed_this_run_seconds': time.perf_counter() - started,
            'cache': str(OUT_PATH.resolve()),
        })

    try:
        save_progress(status)
        with OUT_PATH.open('a', encoding='utf-8') as log:
            for row in tqdm(todo.to_dict('records'), desc=f'OCR {SPLIT}'):
                if RUN_HOURS is not None and time.perf_counter() - started >= RUN_HOURS * 3600:
                    status = 'time_limit'
                    break
                if MAX_IMAGES_PER_RUN is not None and n_new >= MAX_IMAGES_PER_RUN:
                    status = 'image_limit'
                    break
                torch.cuda.synchronize()
                start = time.perf_counter()
                im = oe.spotting_image(DATA_DIR / row['path'])
                try:
                    text, capped, n_tokens = oe.run_spotting(model, proc, im, MAX_NEW_TOKENS)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    result = {
                        'id': row['id'], 'path': row['path'], 'ocr_raw': text,
                        'ocr_tokens': n_tokens, 'ocr_truncated': capped,
                        'max_new_tokens': MAX_NEW_TOKENS, 'prompt': PROMPT,
                        'seconds': seconds, 'image_width': im.width, 'image_height': im.height,
                    }
                    durable_append(log, result)
                    done.add(row['id'])
                    n_new += 1
                    elapsed_total += seconds
                    truncated_count += capped
                finally:
                    im.close()
                if n_new % PROGRESS_EVERY == 0:
                    save_progress('running')
            else:
                status = 'complete'
    except KeyboardInterrupt:
        status = 'interrupted'
        print('사용자 중지. 완료 결과를 확인합니다.')
    except BaseException:
        status = 'error'
        raise
    finally:
        try:
            # The last write might have completed immediately before interruption.
            done = resume_ids(OUT_PATH, MAX_NEW_TOKENS, PROMPT)
            save_progress(status)
            print(f'저장 완료: 대상 중 {len(target_ids & done)}/{len(df)}장 / 상태: {status}')
            print('다음 실행 시 저장된 이미지는 자동으로 건너뜁니다.')
        finally:
            del model, proc
            gc.collect()
            torch.cuda.empty_cache()
    print(f'이번 실행 {n_new}장 / 평균 {elapsed_total / max(n_new, 1):.2f}초')
    print('결과:', OUT_PATH.resolve())
    print('진행 요약:', PROGRESS_PATH.resolve())
''')
check = ''.join(nb['cells'][14]['source'])
check = check.replace('if len(cache) < 6714:\n    print(f"→ test 6,714장 환산: 약 {cache.seconds.mean() * 6714 / 3600:.1f}시간")',
                      'if missing_ids:\n    print(f"남은 {len(missing_ids)}장 예상: 약 {cache.seconds.mean() * len(missing_ids) / 3600:.1f}시간")')
set_cell(14, check)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
        cell['outputs'] = []
        cell['execution_count'] = None
prepare_notebook(nb)
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Updated:', path)
