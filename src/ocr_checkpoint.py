"""Durable, resumable JSONL output for a single local OCR writer."""
import json
import os
import time
from pathlib import Path


def durable_append(log, record):
    log.write(json.dumps(record, ensure_ascii=False) + '\n')
    log.flush()
    os.fsync(log.fileno())


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def resume_ids(path, max_new_tokens, prompt):
    """Validate before mutation; recover torn tails and equivalent duplicates.

    The JSONL is the source of truth, not the optional progress summary.
    Do not run multiple writers against the same path.
    """
    path = Path(path)
    if not path.is_file():
        return set()
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    done = set()
    signatures, kept = {}, []
    duplicate_count = 0
    offset = 0
    repair_at = None
    for index, raw in enumerate(lines):
        if not raw.strip():
            offset += len(raw)
            continue
        try:
            item = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if index == len(lines) - 1 and not raw.endswith(b'\n'):
                repair_at = offset
                break
            raise ValueError(f'캐시 중간 또는 완결된 줄 손상: {path}, 줄 {index + 1}') from error
        if not isinstance(item, dict) or not isinstance(item.get('id'), str):
            raise ValueError(f'캐시 형식 오류: 줄 {index + 1}')
        if not {'ocr_raw', 'ocr_tokens', 'ocr_truncated', 'path'} <= item.keys():
            raise ValueError(f'결과 필드 누락: 줄 {index + 1}')
        if item.get('max_new_tokens') != max_new_tokens or item.get('prompt') != prompt:
            raise ValueError(f'기존 설정과 다릅니다: {item["id"]}. 기존 설정으로 이어 실행하세요.')
        if item['id'] in done:
            signature = {k: v for k, v in item.items()
                         if k not in {'seconds', 'batch_seconds', 'batch_size', 'seconds_kind'}}
            if signature != signatures[item['id']]:
                raise ValueError(f'캐시에 서로 다른 OCR 결과가 중복됨: {item["id"]}. 원본을 확인하세요.')
            duplicate_count += 1
            offset += len(raw)
            continue
        done.add(item['id'])
        signatures[item['id']] = {k: v for k, v in item.items()
                                  if k not in {'seconds', 'batch_seconds', 'batch_size', 'seconds_kind'}}
        kept.append(raw.rstrip(b'\r\n') + b'\n')
        offset += len(raw)
    if repair_at is not None or duplicate_count:
        if path.read_bytes() != data:
            raise RuntimeError('검사 중 캐시가 변경됐습니다. 다른 OCR 실행을 중지하세요.')
        backup = path.with_name(path.name + f'.recovery-{time.time_ns()}.bak')
        with backup.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary = path.with_name(path.name + f'.repair-{time.time_ns()}.tmp')
        with temporary.open('xb') as stream:
            stream.write(b''.join(kept))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, path)
        except PermissionError as error:
            raise PermissionError(
                f'캐시 파일을 교체할 수 없습니다: {path}. 다른 OCR 실행을 중지하고 '
                f'Jupyter/VS Code 커널을 다시 시작한 뒤 실행하세요. '
                f'원본 백업: {backup}; 정리된 임시 파일: {temporary}'
            ) from error
        print(f'동일 결과 중복 {duplicate_count}줄 정리 / 미완성 마지막 줄 {repair_at is not None}. 원본 백업: {backup}')
    elif data and not data.endswith(b'\n'):
        # A valid final JSON record still needs a separator before the next append.
        with path.open('ab') as stream:
            stream.write(b'\n')
            stream.flush()
            os.fsync(stream.fileno())
    return done


def validate_metadata(path, settings):
    path = Path(path)
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding='utf-8'))
    for key, value in settings.items():
        if key == 'rows':
            continue  # LIMIT may change between runs.
        old = previous.get(key)
        if key == 'model':
            old, value = str(old).replace('\\', '/'), str(value).replace('\\', '/')
        if old != value:
            raise ValueError(f'캐시 메타데이터 불일치: {key}: {old!r} -> {value!r}')
