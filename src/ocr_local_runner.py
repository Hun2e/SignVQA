"""Local batched OCR with durable per-record saves and resumable interruption."""
import gc
import time

import torch
from tqdm.auto import tqdm

import ocr_engine as oe
from ocr_checkpoint import atomic_json, durable_append, resume_ids


def infer_rows(model, proc, rows, data_dir, token_cap):
    images = []
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        for row in rows:
            images.append(oe.spotting_image(data_dir / row['path']))
        first_cap = min(512, token_cap)
        outputs = oe.run_spotting_batch(model, proc, images, first_cap)

        # 512토큰에서 잘린 이미지만 1024토큰으로 다시 생성
        retry_indices = [
            i for i, (_, truncated, _) in enumerate(outputs)
            if truncated and first_cap < token_cap
        ]

        if retry_indices:
            retry_outputs = oe.run_spotting_batch(
                model,
                proc,
                [images[i] for i in retry_indices],
                token_cap,
            )
            outputs = list(outputs)
            for i, result in zip(retry_indices, retry_outputs):
                outputs[i] = result

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return [dict(id=row['id'], path=row['path'], ocr_raw=text,
                     ocr_tokens=tokens, ocr_truncated=capped,
                     max_new_tokens=token_cap, prompt=oe.SPOTTING_PROMPT,
                     seconds=elapsed / len(rows), batch_seconds=elapsed,
                     seconds_kind='batch_amortized', batch_size=len(rows),
                     image_width=image.width, image_height=image.height)
                for row, image, (text, capped, tokens) in zip(rows, images, outputs)]
    finally:
        for image in images:
            image.close()


def run_local(df, data_dir, out_path, token_cap, prompt, batch_size=4,
              run_hours=None, max_images=None, progress_every=25):
    assert isinstance(batch_size, int) and batch_size > 0
    assert run_hours is None or run_hours > 0
    assert max_images is None or (isinstance(max_images, int) and max_images > 0)
    assert isinstance(progress_every, int) and progress_every > 0
    done = resume_ids(out_path, token_cap, prompt)
    rows = df[~df.id.isin(done)].to_dict('records')
    if not rows:
        print('이미 모두 완료되었습니다.')
        return
    assert torch.cuda.is_available(), 'CUDA GPU 환경에서 실행하세요.'
    oe.SPOTTING_PROMPT = prompt
    model, proc = oe.load_ocr()
    started = time.perf_counter()
    completed_before = len(set(df.id) & done)
    offset, since_sync = 0, 0
    status = 'running'
    progress_path = out_path.with_suffix('.progress.json')

    def save_progress(state):
        total_done = len(set(df.id) & done)
        atomic_json(progress_path, dict(
            status=state, total=len(df), completed=total_done,
            remaining=len(df) - total_done, new_this_run=total_done - completed_before,
            updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            elapsed_this_run_seconds=time.perf_counter() - started,
            batch_size=batch_size, cache=str(out_path.resolve())))

    try:
        save_progress(status)
        with out_path.open('a', encoding='utf-8') as log, tqdm(total=len(rows), desc='로컬 OCR', unit='장') as bar:
            while offset < len(rows):
                if run_hours is not None and time.perf_counter() - started >= run_hours * 3600:
                    status = 'time_limit'
                    break
                if max_images is not None and offset >= max_images:
                    status = 'image_limit'
                    break
                count = min(batch_size, len(rows) - offset)
                if max_images is not None:
                    count = min(count, max_images - offset)
                oom = False
                try:
                    results = infer_rows(model, proc, rows[offset:offset + count], data_dir, token_cap)
                except torch.cuda.OutOfMemoryError:
                    oom = True
                # Clear traceback-held tensors before retrying the same uncommitted rows.
                if oom:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if count == 1:
                        raise RuntimeError('배치 1에서도 GPU 메모리가 부족합니다. 완료분은 저장됩니다.')
                    batch_size = max(1, count // 2)
                    print(f'GPU 메모리 부족: 배치 {batch_size}로 줄여 재시도합니다.', flush=True)
                    continue
                for record in results:
                    durable_append(log, record)
                    done.add(record['id'])
                    offset += 1
                    since_sync += 1
                    bar.update(1)
                if since_sync >= progress_every:
                    save_progress('running')
                    since_sync = 0
                average = (time.perf_counter() - started) / offset
                bar.set_postfix(batch=batch_size, sec_per_image=round(average, 2),
                                remaining_h=round((len(rows) - offset) * average / 3600, 2))
            else:
                status = 'complete'
    except KeyboardInterrupt:
        status = 'interrupted'
        print('중지 요청: 디스크에 저장된 결과를 확인합니다.')
    except BaseException:
        status = 'error'
        raise
    finally:
        try:
            done = resume_ids(out_path, token_cap, prompt)
            save_progress(status)
            print(f'저장 완료: {len(set(df.id) & done)}/{len(df)}장 / {status}')
            print('결과:', out_path.resolve())
        finally:
            del model, proc
            gc.collect()
            torch.cuda.empty_cache()
