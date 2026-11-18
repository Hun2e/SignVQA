"""E4 이후 실험이 공유하는 Qwen3.5 VLM 적재·추론·짝 비교 코드.

E1~E3는 이 코드를 노트북마다 복사했습니다. E4(27B 제로샷)와 E5(QLoRA)는 E1과
문항 단위로 맞대야 하므로 적재·메시지·생성 규칙을 여기 한 벌만 둡니다.
프롬프트와 파서는 여전히 common_eval에서 옵니다.
"""
import json
import math
import time
from math import comb
from pathlib import Path

import pandas as pd
import torch

import common_eval as ce

E1_PRED = Path("output/exp_E1_qwen35_9b/predictions.jsonl")
E1_TAXONOMY = Path("output/exp_E1_qwen35_9b/error_taxonomy.csv")


def check_weights(model_dir):
    """config와 index에 적힌 shard가 전부 있는지 확인합니다. 다운로드 중단을 잡습니다."""
    model_dir = Path(model_dir)
    assert (model_dir / "config.json").is_file(), f"로컬 모델을 준비하세요: {model_dir}"
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        need = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        miss = [f for f in need if not (model_dir / f).is_file()]
        assert not miss, f"가중치 shard {len(miss)}/{len(need)}개 누락: {miss[:2]}"
    assert list(model_dir.glob("*.safetensors")), f"가중치(.safetensors)가 없습니다: {model_dir}"


def build_device_map(model_dir, cpu_modules=(), cpu_last_layers=0):
    """GPU에 다 못 올릴 때 CPU로 보낼 모듈을 명시합니다.

    device_map="auto"는 앞에서부터 채우므로 비전 인코더나 앞쪽 층이 CPU로 밀릴 수 있습니다.
    27B는 임베딩과 lm_head(각 약 2.5 GiB, 양자화 대상 아님)만 빼면 4bit 본체가 16 GiB에 들어가므로
    그 둘을 먼저 내리고, 그래도 부족할 때만 마지막 층부터 내립니다.
    """
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    n_layers = cfg["text_config"]["num_hidden_layers"]
    lm = "model.language_model"
    dmap = {"model.visual": 0, f"{lm}.embed_tokens": 0, f"{lm}.norm": 0,
            f"{lm}.rotary_emb": 0, "lm_head": 0}
    for i in range(n_layers):
        dmap[f"{lm}.layers.{i}"] = "cpu" if i >= n_layers - cpu_last_layers else 0
    for name in cpu_modules:
        assert name in dmap, f"알 수 없는 모듈: {name}"
        dmap[name] = "cpu"
    return dmap


def execute_on_cpu(module, out_device="cuda:0"):
    """accelerate 오프로딩 훅을 떼고 모듈을 CPU에서 직접 계산하게 바꿉니다.

    기본 오프로딩은 forward마다 가중치를 GPU로 복사합니다. 27B의 임베딩·lm_head는 합쳐 5 GiB라
    문항당 수 초가 복사에 들어갑니다. 임베딩은 조회뿐이고 lm_head는 생성 중 마지막 토큰 하나만
    계산하므로 CPU에서 돌리고 결과만 GPU로 보내는 편이 훨씬 빠릅니다.
    """
    from accelerate.hooks import remove_hook_from_module
    from accelerate.utils import set_module_tensor_to_device

    hook = getattr(module, "_hf_hook", None)
    weights = getattr(hook, "weights_map", None)
    names = [n for n, _ in module.named_parameters()]
    values = {n: weights[n] for n in names} if weights is not None else {}
    remove_hook_from_module(module, recurse=True)
    for n in names:
        if n in values:
            set_module_tensor_to_device(module, n, "cpu", value=values[n])
    assert all(p.device.type == "cpu" for p in module.parameters()), "CPU 적재 실패"
    module.register_forward_pre_hook(
        lambda m, args, kwargs: (tuple(a.to("cpu") if torch.is_tensor(a) else a for a in args),
                                 {k: v.to("cpu") if torch.is_tensor(v) else v
                                  for k, v in kwargs.items()}), with_kwargs=True)
    module.register_forward_hook(lambda m, args, out: out.to(out_device))


def load_vlm(model_dir, four_bit=True, dtype=torch.bfloat16, cpu_modules=(), cpu_last_layers=0,
             adapter_dir=None):
    """E1과 같은 nf4 설정으로 적재합니다. adapter_dir를 주면 LoRA를 얹습니다."""
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    check_weights(model_dir)
    proc = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    offload = bool(cpu_modules) or cpu_last_layers > 0
    dmap = (build_device_map(model_dir, cpu_modules, cpu_last_layers) if offload else {"": 0})
    kwargs = dict(local_files_only=True, dtype=dtype, device_map=dmap)
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            # CPU로 보낸 모듈은 양자화하지 않고 dtype 그대로 둡니다.
            llm_int8_enable_fp32_cpu_offload=offload)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(model_dir), **kwargs)
    for name in cpu_modules:
        execute_on_cpu(model.get_submodule(name))
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    return model.eval(), proc


def build_messages(row, image, consensus=False):
    """E1과 글자 단위로 같은 메시지. 학습(E5·E7)도 이 함수로 입력을 만들어야 분포가 맞습니다.

    consensus=True면 마지막 지시문만 E6의 합의 지시문으로 바뀝니다.
    """
    return [{"role": "system", "content": [{"type": "text", "text": ce.SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text",
                                          "text": ce.make_prompt(row, consensus=consensus)}]}]


def encode(proc, messages, enable_thinking=False):
    return proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                    return_dict=True, return_tensors="pt",
                                    enable_thinking=enable_thinking)


def letter_token_ids(proc):
    """a~d 각각이 첫 토큰이 될 수 있는 id 후보. 대소문자·앞 공백 변형을 모읍니다."""
    tok = proc.tokenizer
    table = {}
    for ch in ce.CHOICES:
        cand = set()
        for form in (ch, " " + ch, ch.upper(), " " + ch.upper()):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                cand.add(int(enc[0]))
        assert cand, f"'{ch}'의 토큰 id를 찾지 못했습니다"
        table[ch] = sorted(cand)
    return table


@torch.inference_mode()
def answer(model, proc, inputs, limit, letters, device="cuda:0"):
    """E1과 같은 탐욕 생성에 더해, 첫 토큰의 a~d 로그확률을 함께 돌려줍니다.

    예측은 생성 결과를 parse_choice로 읽은 값 그대로이고(E1과 동일), 로그확률은
    나중에 모델 앙상블·선지 순서 TTA에 쓰려고 추가 비용 없이 남기는 기록입니다.
    """
    inputs = inputs.to(device)
    prefix_len = inputs["input_ids"].shape[-1]
    out = model.generate(**inputs, max_new_tokens=limit, do_sample=False, use_cache=True,
                         output_logits=True, return_dict_in_generate=True)
    new = out.sequences[0, prefix_len:]
    eos = model.generation_config.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    truncated = len(new) >= limit and (not len(new) or int(new[-1]) not in eos_ids)
    logp = torch.log_softmax(out.logits[0][0].float(), dim=-1)
    letter_logp = {ch: max(float(logp[i]) for i in ids) for ch, ids in letters.items()}
    raw = proc.decode(new, skip_special_tokens=True).strip()
    return raw, bool(truncated), int(len(new)), letter_logp


def run_predictions(model, proc, rows, pred_path, limit=24, resume=True, desc="predict"):
    """문항마다 predictions.jsonl에 기록하고 flush합니다. 다시 부르면 끝난 문항을 건너뜁니다."""
    from tqdm.auto import tqdm

    pred_path = Path(pred_path)
    done = set()
    if resume and pred_path.is_file():
        with pred_path.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
        print(f"이어서 진행: 이미 끝난 {len(done)}개 건너뜀")
    elif not resume and pred_path.is_file():
        pred_path.unlink()
    todo = [r for r in rows if r["id"] not in done]
    print(f"남은 {len(todo)}개")

    letters = letter_token_ids(proc)
    torch.cuda.reset_peak_memory_stats()
    with pred_path.open("a", encoding="utf-8") as log:
        for row in tqdm(todo, desc=desc):
            torch.cuda.synchronize(); start = time.perf_counter()
            im = ce.open_image(row)
            inputs = encode(proc, build_messages(row, im))
            raw, capped, n_tok, letter_logp = answer(model, proc, inputs, limit, letters)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            log.write(json.dumps(dict(
                id=row["id"], raw_output=raw, pred=ce.parse_choice(raw), letter_logprobs=letter_logp,
                answer_truncated=capped, answer_tokens=n_tok, seconds=elapsed,
                image_width=im.width, image_height=im.height), ensure_ascii=False) + "\n")
            log.flush()
            del inputs, im
    return torch.cuda.max_memory_allocated() / 1024**3


def read_predictions(pred_path, val_df):
    res = pd.read_json(pred_path, lines=True).drop_duplicates("id", keep="last")
    pred_df = val_df.merge(res, on="id", validate="one_to_one", sort=False)
    pred_df["correct"] = pred_df.pred == pred_df.answer
    return pred_df


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def paired_vs_e1(pred_df, name, e1_pred=E1_PRED, taxonomy=E1_TAXONOMY):
    """E1과 문항 단위로 맞댑니다. 정확도 두 개를 나란히 놓는 것만으로는 부족합니다.

    E1 오답 분류(error_taxonomy.csv)가 있으면 분류별로 몇 개를 고쳤는지도 봅니다.
    27B는 판독·위치 관계를, QLoRA는 '적힌 그대로' 유형을 고칠 것이라는 예상을 여기서 확인합니다.
    """
    if not Path(e1_pred).is_file():
        print("E1 예측이 없어 짝 비교를 건너뜁니다:", e1_pred)
        return None
    e1 = pd.read_json(e1_pred, lines=True).drop_duplicates("id", keep="last")
    joined = pred_df.merge(e1[["id", "pred"]].rename(columns={"pred": "pred_e1"}), on="id")
    ok1 = joined.pred_e1 == joined.answer
    ok2 = joined.pred == joined.answer
    n = len(joined)
    b = int((ok1 & ~ok2).sum())
    c = int((~ok1 & ok2).sum())
    diff = (c - b) / n
    se = math.sqrt(b + c) / n
    p = mcnemar_exact(b, c)
    print(f"[{n}문항]")
    print(f"  E1   {ok1.mean():.4f} ({int(ok1.sum())}/{n})")
    print(f"  {name} {ok2.mean():.4f} ({int(ok2.sum())}/{n})")
    print(f"  E1만 정답 {b} / {name}만 정답 {c} / 둘 다 오답 {int((~ok1 & ~ok2).sum())}")
    print(f"  {name} - E1 = {diff * 100:+.2f}pp  95%CI "
          f"[{(diff - 1.96 * se) * 100:+.2f}pp, {(diff + 1.96 * se) * 100:+.2f}pp]")
    print(f"  McNemar exact p = {p:.4g} -> "
          + ("차이가 유의합니다" if p < 0.05 else "두 조건은 구분되지 않습니다"))

    if Path(taxonomy).is_file():
        tax = pd.read_csv(taxonomy, encoding="utf-8")
        t = tax.merge(joined[["id", "pred", "answer"]], on="id", how="left")
        t["fixed"] = t.pred == t.answer
        table = t.groupby("category", sort=False).agg(E1_errors=("id", "size"),
                                                      fixed=("fixed", "sum"))
        print("\nE1 오답 분류별로 고친 수 (label_error·ambiguous는 고쳐도 운에 가깝습니다)")
        print(table.to_string())
    return dict(n=n, e1_only=b, new_only=c, diff=diff, p=p)


def _self_test():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(11, 11) == 1.0
    assert abs(mcnemar_exact(11, 17) - 0.3449) < 1e-3   # E2 vs E1 기록값
    row = {"question": "Q", "a": "A", "b": "B", "c": "C", "d": "D"}
    assert build_messages(row, None)[1]["content"][1]["text"] == ce.make_prompt(row)
    print("vlm_runtime OK")


if __name__ == "__main__":
    _self_test()
