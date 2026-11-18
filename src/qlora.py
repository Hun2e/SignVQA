"""E5 QLoRA 학습 코드.

핵심 규칙은 하나입니다. **학습 입력은 추론 입력과 토큰 단위로 같아야 합니다.**
그래서 프롬프트는 vlm_runtime.build_messages + encode(add_generation_prompt=True)로 만들고,
그 뒤에 정답 글자와 <|im_end|>만 붙입니다. 채팅 템플릿에 assistant 메시지를 넣어 렌더링하면
thinking 블록 처리가 추론 때와 달라질 수 있어 쓰지 않습니다.
"""
import json
import math
from pathlib import Path

import pandas as pd
import torch

import common_eval as ce
import vlm_runtime as vr

# 언어 모델 쪽 선형층만. 비전 인코더와 merger는 얼립니다(E1 오답의 다수가 '읽고 나서 고치는' 쪽이라
# 우선 언어 쪽 습관을 바꿔 보고, 판독 실패가 안 줄면 그때 merger를 엽니다).
LORA_TARGET = (r"model\.language_model\.layers\.\d+\."
               r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
               r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
               r"|mlp\.(gate_proj|up_proj|down_proj))")


def load_train_rows(val_df, train_csv=ce.DATA_DIR / "train.csv", seed=42):
    """train.csv에서 검증 672문항을 뺀 나머지 전부. 검증과 ID·이미지가 겹치지 않는지 확인합니다."""
    train = pd.read_csv(train_csv, keep_default_na=False)
    val_ids, val_paths = set(val_df.id), set(val_df.path)
    fit = train[~train.id.isin(val_ids)].copy()
    assert not set(fit.path) & val_paths, "학습 이미지가 검증과 겹칩니다"
    assert len(fit) + len(val_ids) == len(train), "검증 ID 일부가 train.csv에 없습니다"
    assert set(fit.answer) <= set(ce.CHOICES), "정답 형식이 a~d가 아닙니다"
    return fit.sample(frac=1.0, random_state=seed).reset_index(drop=True)


DEV_VOTES = [f"answer{i}" for i in range(1, 6)]


def dev_majority(row, min_votes=3):
    """dev의 5명 응답 중 min_votes표 이상 받은 단독 최다 답. 없으면 None."""
    votes = [row[c] for c in DEV_VOTES if row.get(c) in ce.CHOICES]
    if not votes:
        return None
    counts = {ch: votes.count(ch) for ch in set(votes)}
    best = max(counts.values())
    leaders = [ch for ch, k in counts.items() if k == best]
    return leaders[0] if best >= min_votes and len(leaders) == 1 else None


def rows_from_manifest(manifest, train_df, dev_df, holdout_ids, min_votes=3):
    """학습 구성 목록(id, source)을 학습 행으로 바꿉니다. 정답 글자는 저장소에 두지 않고 여기서 붙입니다.

    train 문항은 train.csv의 정답을, dev 문항은 다수결(min_votes표 이상, 단독 최다)을 씁니다.
    holdout 문항이 섞이면 평가가 무의미해지므로 멈춥니다.
    """
    assert set(manifest.source) <= {"train", "dev"}, f"알 수 없는 source: {set(manifest.source)}"
    leaked = set(manifest.id) & set(holdout_ids)
    assert not leaked, f"holdout 문항 {len(leaked)}개가 학습 구성에 들어 있습니다: {sorted(leaked)[:3]}"
    tr = train_df.set_index("id")
    dv = dev_df.set_index("id")
    rows, dropped = [], 0
    for item in manifest.itertuples():
        if item.source == "train":
            rows.append({"id": item.id, **tr.loc[item.id].to_dict()})
        else:
            r = dv.loc[item.id].to_dict()
            answer = dev_majority(r, min_votes)
            if answer is None:
                dropped += 1
                continue
            rows.append({"id": item.id, **{k: v for k, v in r.items() if k not in DEV_VOTES},
                         "answer": answer})
    if dropped:
        print(f"dev 다수결이 {min_votes}표에 못 미치거나 동률이라 뺀 문항 {dropped}개")
    return rows


def target_ids(proc, letter):
    tok = proc.tokenizer
    ids = tok.encode(letter, add_special_tokens=False)
    assert len(ids) == 1, f"'{letter}'가 한 토큰이 아닙니다: {ids}"
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert isinstance(im_end, int) and im_end >= 0
    return ids + [im_end]


def make_example(proc, row, image, consensus=False):
    """추론과 같은 프롬프트 + [정답 글자, <|im_end|>]. 손실은 뒤의 두 토큰에만 겁니다."""
    inputs = vr.encode(proc, vr.build_messages(row, image, consensus=consensus))
    tgt = torch.tensor([target_ids(proc, row["answer"])], dtype=inputs["input_ids"].dtype)
    n_prompt = inputs["input_ids"].shape[-1]
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.shape == (1, n_prompt):
            fill = torch.ones_like(tgt) if k == "attention_mask" else (
                tgt if k == "input_ids" else torch.zeros_like(tgt))
            out[k] = torch.cat([v, fill.to(v.dtype)], dim=1)
        else:
            out[k] = v
    return out, tgt


def answer_loss(model, batch, tgt, device="cuda:0"):
    """정답 두 토큰에 대한 CE. 마지막 n+1개 위치의 로짓만 계산해 248K 어휘 전체 로짓을 피합니다.

    위치 p의 로짓은 p+1 토큰을 예측하므로, 끝에서 n+1개를 남기면 앞의 n개가 정답 토큰을 예측합니다.
    """
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    tgt = tgt.to(device)
    n = tgt.shape[-1]
    logits = model(**batch, logits_to_keep=n + 1, use_cache=False).logits[:, :n].float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))


def attach_lora(model, r=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=LORA_TARGET, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_train > 0, "LoRA가 붙은 층이 없습니다. LORA_TARGET을 확인하세요"
    return model


def cosine_lr(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def save_checkpoint(model, optimizer, scheduler, state, ckpt_dir):
    """어댑터 + 옵티마이저 + 진행 위치. Windows 장시간 학습이 끊겨도 이어갈 수 있게 합니다."""
    ckpt_dir = Path(ckpt_dir)
    tmp = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
    model.save_pretrained(str(tmp))
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
               tmp / "train_state.pt")
    (tmp / "progress.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    if ckpt_dir.exists():
        import shutil
        shutil.rmtree(ckpt_dir)
    tmp.rename(ckpt_dir)


def load_checkpoint(model, optimizer, scheduler, ckpt_dir):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    ckpt_dir = Path(ckpt_dir)
    set_peft_model_state_dict(model, load_file(str(ckpt_dir / "adapter_model.safetensors")))
    st = torch.load(ckpt_dir / "train_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(st["optimizer"])
    scheduler.load_state_dict(st["scheduler"])
    return json.loads((ckpt_dir / "progress.json").read_text(encoding="utf-8"))


def train_loop(model, proc, order, *, grad_accum, lr, warmup_ratio, max_grad_norm, save_every,
               ckpt_dir, log_path, consensus=False, resume=True):
    """문항 하나씩 손실을 쌓아 grad_accum마다 갱신합니다. save_every 스텝마다 ckpt_dir에 저장하고,
    다시 부르면 마지막 체크포인트에서 이어 갑니다. Colab 세션이 끊겨도 Drive의 ckpt_dir로 재개합니다.
    """
    import time
    import common_eval as _ce

    total = len(order) // grad_accum
    warmup = max(1, int(total * warmup_ratio))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_lr(s, total, warmup))
    ckpt_dir, log_path = Path(ckpt_dir), Path(log_path)
    start = 0
    if resume and (ckpt_dir / "progress.json").is_file():
        start = load_checkpoint(model, optimizer, scheduler, ckpt_dir)["step"]
        print(f"체크포인트에서 이어서: {start}/{total} 스텝")
    elif log_path.is_file():
        log_path.unlink()
    model.train()
    t_start = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        for step in range(start, total):
            t0 = time.time()
            losses = []
            for row in order[step * grad_accum:(step + 1) * grad_accum]:
                batch, tgt = make_example(proc, row, _ce.open_image(row), consensus=consensus)
                loss = answer_loss(model, batch, tgt)
                (loss / grad_accum).backward()
                losses.append(loss.item())
                del batch, tgt, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            item = dict(step=step + 1, loss=sum(losses) / len(losses), max_loss=max(losses),
                        lr=scheduler.get_last_lr()[0], grad_norm=float(grad_norm),
                        seconds=time.time() - t0)
            log.write(json.dumps(item) + "\n"); log.flush()
            done = step + 1 - start
            if done == 1 or (step + 1) % 10 == 0 or step + 1 == total:
                left = (time.time() - t_start) / done * (total - step - 1)
                print(f"  {step + 1}/{total}  loss {item['loss']:.4f}  max {item['max_loss']:.3f}  "
                      f"{item['seconds']:.0f}초/스텝  남은 약 {left / 60:.0f}분", flush=True)
            if (step + 1) % save_every == 0 or step + 1 == total:
                save_checkpoint(model, optimizer, scheduler, {"step": step + 1, "total": total}, ckpt_dir)
    return total


def merge_and_save(base_dir, adapter_dir, out_dir, dtype=torch.bfloat16, device="cuda:0"):
    """원본(bf16)에 어댑터를 합쳐 일반 체크포인트로 저장합니다. vLLM은 이 폴더를 그대로 올립니다.

    vLLM이 Qwen3.5 linear attention 층의 LoRA를 직접 받는지 보장할 수 없어 합친 모델로 평가합니다.
    4bit로 학습했어도 합치는 대상은 bf16 원본입니다.
    """
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(base_dir), dtype=dtype, device_map={"": device}, local_files_only=True)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    out_dir = Path(out_dir)
    merged.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(str(base_dir), local_files_only=True).save_pretrained(str(out_dir))
    del merged, base
    copy_missing_weights(base_dir, out_dir)
    return out_dir


def copy_missing_weights(base_dir, out_dir):
    """원본에는 있는데 저장본에 없는 가중치(mtp.* 다중 토큰 예측 층)를 그대로 옮겨 담습니다.

    transformers는 mtp 층을 불러오지 않아 save_pretrained 결과에서 빠집니다. 학습과 무관한 층이지만
    config에는 남아 있어, vLLM이 찾을 경우를 대비해 원본과 같은 가중치 집합으로 맞춥니다.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_dir, out_dir = Path(base_dir), Path(out_dir)
    base_map = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    index_path = out_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    missing = sorted(set(base_map) - set(index["weight_map"]))
    assert all(k.startswith("mtp.") for k in missing), f"예상 밖 누락: {[k for k in missing if not k.startswith('mtp.')][:3]}"
    if not missing:
        return []
    tensors = {}
    for shard in sorted({base_map[k] for k in missing}):
        with safe_open(str(base_dir / shard), framework="pt") as f:
            for k in missing:
                if base_map[k] == shard:
                    tensors[k] = f.get_tensor(k)
    save_file(tensors, str(out_dir / "model-extra-mtp.safetensors"), metadata={"format": "pt"})
    for k in missing:
        index["weight_map"][k] = "model-extra-mtp.safetensors"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return missing


def _self_test():
    import re
    ok = "model.language_model.layers.3.linear_attn.in_proj_qkv"
    assert re.fullmatch(LORA_TARGET, ok)
    assert re.fullmatch(LORA_TARGET, "model.language_model.layers.7.mlp.down_proj")
    assert not re.fullmatch(LORA_TARGET, "model.visual.blocks.0.attn.qkv")
    assert not re.fullmatch(LORA_TARGET, "lm_head")
    assert abs(cosine_lr(0, 100, 10) - 0.1) < 1e-9 and abs(cosine_lr(100, 100, 10)) < 1e-9
    votes = lambda *v: dict(zip(DEV_VOTES, list(v) + [""] * (5 - len(v))))
    assert dev_majority(votes("a", "a", "a", "b", "c")) == "a"
    assert dev_majority(votes("a", "a", "b", "b", "c")) is None        # 2표 동률
    assert dev_majority(votes("a", "a", "b", "c", "d")) is None        # 2표 단독이지만 3표 미만
    assert dev_majority(votes("a", "a", "b", "c", "d"), min_votes=2) == "a"
    assert dev_majority(votes()) is None
    print("qlora OK")


if __name__ == "__main__":
    _self_test()
