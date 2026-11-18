"""Create the standalone Colab batch OCR notebook from the existing notebook."""

from _project import ROOT
import ast
import json
from pathlib import Path

root = ROOT / 'notebooks/ocr'
nb = json.loads((root / 'AI_Challenge2_OCR_cache_colab.ipynb').read_text(encoding='utf-8'))

def source(index, text):
    nb['cells'][index]['source'] = text.strip().splitlines(keepends=True)

source(0, '''# A100 OCR 캐시 생성 — 배치 크기 자동 비교

기존 Colab 노트북을 **정상 중지하여 Drive 저장을 마친 뒤** 이 파일을 실행하세요.
같은 캐시에 두 노트북을 동시에 실행하지 마세요. 기존 Spotting 캐시를 이어서 처리합니다.

1. A100 런타임에서 설치 셀 실행 → 필요하면 세션 재시작 → 다음 셀부터 실행.
2. `ZIP_PATH`, `DRIVE_BASE`를 자신의 Drive 경로에 맞춥니다.
3. 실행 셀은 남은 데이터 중 고정된 최대 16장으로 배치 1/2/4/8을 비교합니다.
4. 단일 처리와 샘플 출력이 모두 같은 후보 중 가장 빠른 크기로 전체를 처리합니다.

측정에는 원래 1024토큰 한도를 사용합니다. 벤치마크는 몇 분 걸릴 수 있습니다.
샘플 일치는 전체 결과 동일성이나 정확도 보장이 아닙니다. GPU의 수치 연산 차이로
배치 결과가 달라지는 후보는 보수적으로 제외합니다. 속도 개선이 없으면 배치 1을 씁니다.
해상도, 프롬프트, 토큰 한도는 원래 설정을 유지합니다.

공식 배치 처리 안내: https://huggingface.co/docs/transformers/en/model_doc/paddleocr_vl#batched-inference
''')

# Avoid traversing mounted Drive just to discover local data.
source(6, '''import os
from pathlib import Path

ZIP_PATH = "/content/drive/MyDrive/SSAFY/AI_Challenge2/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "ocrisnotenough"

def local_csv_files(name):
    for parent, dirs, files in os.walk('/content'):
        dirs[:] = [d for d in dirs if d not in {'drive', '.cache', 'sample_data'}]
        if name in files:
            yield Path(parent) / name

assert os.path.isfile(ZIP_PATH), f"zip을 찾지 못했습니다: {ZIP_PATH}"
if list(local_csv_files('train.csv')):
    print('로컬 데이터가 있어 압축 해제를 건너뜁니다.')
else:
    !7z x -y -p"$ZIP_PASSWORD" -o/content/ "$ZIP_PATH"
print('로컬 train.csv:', list(local_csv_files('train.csv')))
''')
imports = ''.join(nb['cells'][8]['source'])
source(8, imports + '\nassert torch.cuda.is_available(), "GPU 런타임을 선택하세요."\nprint("GPU:", torch.cuda.get_device_name(0))\n')
params = ''.join(nb['cells'][10]['source'])
source(10, params + '''

BENCHMARK_SAMPLES = 16
BATCH_CANDIDATES = [1, 2, 4, 8]
''')
engine = ''.join(nb['cells'][12]['source'])
engine = engine.replace('    configure_ocr_pixels(proc)\n', '    proc.tokenizer.padding_side = "left"\n    configure_ocr_pixels(proc)\n')
source(12, engine + '''

@torch.inference_mode()
def run_batch(model, proc, rows):
    images = []
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        images = [spotting_image(DATA_DIR / row['path']) for row in rows]
        messages = [[{'role': 'user', 'content': [
            {'type': 'image', 'image': im}, {'type': 'text', 'text': PROMPT}]}]
            for im in images]
        import inspect
        padding_kwargs = ({'processor_kwargs': {'padding': True}}
                          if 'processor_kwargs' in inspect.signature(proc.apply_chat_template).parameters
                          else {'padding': True})
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **padding_kwargs,
            return_dict=True, return_tensors='pt').to(model.device)
        prefix_len = inputs['input_ids'].shape[-1]
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False, use_cache=True)
        eos = model.generation_config.eos_token_id
        eos_ids = set([] if eos is None else ([eos] if isinstance(eos, int) else eos))
        results = []
        # Completed rows are padded until the longest row finishes. Strip at first EOS.
        for row, im, sequence in zip(rows, images, output[:, prefix_len:].tolist()):
            end = next((i + 1 for i, token in enumerate(sequence) if token in eos_ids), None)
            tokens = sequence if end is None else sequence[:end]
            results.append({
                'id': row['id'], 'path': row['path'],
                'ocr_raw': proc.decode(tokens, skip_special_tokens=True).strip(),
                'ocr_tokens': len(tokens),
                'ocr_truncated': end is None and len(tokens) >= MAX_NEW_TOKENS,
                'max_new_tokens': MAX_NEW_TOKENS, 'prompt': PROMPT,
                'image_width': im.width, 'image_height': im.height,
                'batch_size': len(rows), 'seconds_kind': 'batch_amortized',
            })
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        for result in results:
            result['seconds'] = elapsed / len(rows)
            result['batch_seconds'] = elapsed
        return results
    finally:
        for im in images:
            im.close()

def clear_oom():
    gc.collect()
    torch.cuda.empty_cache()

def signature(rows):
    return [(r['ocr_raw'], r['ocr_tokens'], r['ocr_truncated']) for r in rows]

def choose_batch_size(model, proc, todo):
    sample = todo.sample(n=min(BENCHMARK_SAMPLES, len(todo)), random_state=42).to_dict('records')
    print('벤치마크 워밍업 중...', flush=True)
    run_batch(model, proc, sample[:1])  # warm-up, excluded from timing
    reference, reports = None, []
    for batch_size in sorted(set([1] + BATCH_CANDIDATES)):
        if batch_size > len(sample):
            continue
        clear_oom()
        torch.cuda.reset_peak_memory_stats()
        outputs = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        oom = False
        try:
            for offset in tqdm(range(0, len(sample), batch_size),
                               desc=f'벤치마크 batch={batch_size}', unit='batch'):
                outputs.extend(run_batch(model, proc, sample[offset:offset + batch_size]))
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            clear_oom()
            if batch_size == 1:
                raise RuntimeError('배치 1도 GPU 메모리가 부족합니다. 다른 모델을 해제하세요.')
            print(f'배치 {batch_size}: OOM — 제외')
            continue
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if reference is None:
            reference = signature(outputs)
        exact = signature(outputs) == reference
        report = dict(batch_size=batch_size, seconds_per_image=elapsed / len(sample),
                      peak_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      matches_single=exact)
        reports.append(report)
        print(report, flush=True)
    best = min((r for r in reports if r['matches_single']), key=lambda r: r['seconds_per_image'])
    report_path = DRIVE_CACHE / f'{SPLIT}_a100_benchmark.json'
    report_path.write_text(json.dumps({
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'transformers': __import__('transformers').__version__,
        'sample_ids': [r['id'] for r in sample], 'results': reports,
        'selected_batch_size': best['batch_size'],
        'model': MODEL_ID, 'prompt': PROMPT, 'max_new_tokens': MAX_NEW_TOKENS,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"선택: 배치 {best['batch_size']} / 샘플 기준 남은 작업 약 "
          f"{len(todo) * best['seconds_per_image'] / 3600:.2f}시간 (추정)")
    return best['batch_size']
''')
data = ''.join(nb['cells'][14]['source']).replace('Path("/content").rglob(f"{split}.csv")', 'local_csv_files(f"{split}.csv")')
source(14, data)
source(15, '''## 7. 이어하기 준비

로컬과 Drive 캐시를 모두 읽어 완료 ID를 합칩니다. 설정이나 동일 ID의 결과가 충돌하면 중단합니다.
파일 중간 손상은 중단하고, 끊긴 마지막 줄만 제외합니다.
''')
source(16, '''def read_cache(path):
    records = {}
    if not path.is_file():
        return records
    lines = path.read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            key = item['id']
        except (json.JSONDecodeError, KeyError):
            if index != len(lines) - 1:
                raise ValueError(f'캐시 중간 손상: {path}, 줄 {index + 1}')
            print('중단으로 손상된 마지막 줄 제외:', path)
            continue
        if item.get('max_new_tokens') != MAX_NEW_TOKENS or item.get('prompt') != PROMPT:
            raise ValueError(f'캐시 설정 불일치: {path}, {key}')
        if key in records and records[key] != item:
            raise ValueError(f'같은 ID의 서로 다른 결과: {key}')
        records[key] = item
    return records

# Read both before writing: resume also works when either local or Drive has more rows.
records = read_cache(DRIVE_PATH)
for key, item in read_cache(LOCAL_PATH).items():
    if key in records and records[key] != item:
        raise ValueError(f'로컬/Drive 결과 충돌: {key}. 원본을 확인하세요.')
    records[key] = item
if records:
    temp = LOCAL_PATH.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\\n'
                            for r in records.values()), encoding='utf-8')
    temp.replace(LOCAL_PATH)
done = set(records)
todo = df[~df.id.isin(done)]
print(f'전체 {len(df)}장 / 대상 중 완료 {len(df) - len(todo)}장 / 남음 {len(todo)}장')
del records
''')
source(17, '''## 8. 벤치마크 → 전체 OCR

단일 처리와 샘플 결과가 일치하는 가장 빠른 배치를 선택합니다.
실행 중 메모리가 부족하면 해당 미완료 배치를 절반으로 줄여 재시도합니다.
Ctrl+C/중지 시 완료분을 Drive에 저장합니다. 강제 런타임 종료는 마지막 동기화 이후 결과를 잃을 수 있습니다.
배치 결과의 `seconds`는 개별 지연 시간이 아니라 **배치 시간 / 이미지 수**입니다.
''')
source(18, '''def sync_to_drive():
    if LOCAL_PATH.is_file():
        temp = DRIVE_PATH.with_suffix('.jsonl.tmp')
        shutil.copy(LOCAL_PATH, temp)
        temp.replace(DRIVE_PATH)

# Recompute on every execution, including when this cell is rerun after interruption.
done = set(read_cache(LOCAL_PATH))
todo = df[~df.id.isin(done)]
if todo.empty:
    print('이미 모두 완료되었습니다.')
else:
    model, proc = load_ocr()
    try:
        batch_size = choose_batch_size(model, proc, todo)
        torch.cuda.reset_peak_memory_stats()
        rows = todo.to_dict('records')
        offset, since_sync = 0, 0
        started = time.perf_counter()
        with LOCAL_PATH.open('a', encoding='utf-8') as log, tqdm(total=len(rows), desc=f'OCR {SPLIT}') as bar:
            while offset < len(rows):
                current = rows[offset:offset + batch_size]
                oom = False
                try:
                    results = run_batch(model, proc, current)
                except torch.cuda.OutOfMemoryError:
                    oom = True
                if oom:
                    clear_oom()
                    if batch_size == 1:
                        raise RuntimeError('배치 1에서 OOM. 완료분 저장 후 중단합니다.')
                    batch_size = max(1, batch_size // 2)
                    print('메모리 부족: 배치를 줄여 재시도:', batch_size)
                    continue
                for result in results:
                    log.write(json.dumps(result, ensure_ascii=False) + '\\n')
                log.flush()
                offset += len(results)
                since_sync += len(results)
                bar.update(len(results))
                if since_sync >= SYNC_EVERY:
                    sync_to_drive()
                    since_sync = 0
                seconds_per_image = (time.perf_counter() - started) / offset
                bar.set_postfix(batch=batch_size, sec_per_image=round(seconds_per_image, 2),
                                remaining_h=round((len(rows) - offset) * seconds_per_image / 3600, 2))
        print(f'완료 {offset}장 / 본 처리 {(time.perf_counter() - started) / 3600:.2f}시간')
        print(f'GPU peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB')
    finally:
        try:
            sync_to_drive()
        finally:
            del model, proc
            clear_oom()
    print('Drive 저장:', DRIVE_PATH)
''')
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
        # IPython shell escapes are parsed separately by IPython, not Python ast.
        code = ''.join(cell['source'])
        ast.parse('\n'.join(line[:len(line) - len(line.lstrip())] + 'pass'
                            if line.lstrip().startswith('!') else line for line in code.splitlines()))
out = root / 'AI_Challenge2_OCR_cache_colab_A100.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(out)
