"""Colab vLLM 추론 공통 코드 (E7 이후).

E4·E6 Colab 노트북은 이 코드를 셀에 복사해 두었습니다. E7부터는 노트북이 이 파일 원문을 품고
import합니다. 메시지·이미지 규칙·파서는 로컬 추론(vlm_runtime)과 같아야 하므로 common_eval을 씁니다.
"""
import json
import math
import shutil
import time
from math import comb
from pathlib import Path

import pandas as pd

import common_eval as ce

CHUNK = 256


def letter_token_ids(tokenizer):
    return {ch: sorted({tokenizer.encode(f, add_special_tokens=False)[0]
                        for f in (ch, " " + ch, ch.upper(), " " + ch.upper())
                        if tokenizer.encode(f, add_special_tokens=False)})
            for ch in ce.CHOICES}


def make_request(proc, row, consensus=False, image_fn=None):
    """로컬과 같은 메시지를 HF 채팅 템플릿으로 렌더링하고, 이미지는 같은 open_image로 줄여 넘깁니다.

    image_fn을 주면 open_image 대신 씁니다(E8 Zoom의 고해상도 입력).
    """
    image = (image_fn or ce.open_image)(row)
    messages = [{"role": "system", "content": [{"type": "text", "text": ce.SYSTEM}]},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text",
                                              "text": ce.make_prompt(row, consensus=consensus)}]}]
    prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    return {"prompt": prompt, "multi_modal_data": {"image": image}}, image.size


def letter_logprobs(first_step, letters):
    """상위 k개 안에 없는 글자는 -1e9. JSON에 -inf를 쓰지 않으려고 유한값을 씁니다."""
    out = {}
    for ch, ids in letters.items():
        vals = [first_step[i].logprob for i in ids if i in first_step]
        out[ch] = max(vals) if vals else -1e9
    return out


def run(llm, proc, sampling, rows, out_dir, drive_dir=None, consensus=False, desc="", image_fn=None):
    """묶음 단위로 추론해 predictions.jsonl에 쓰고, 묶음마다 drive_dir로 복사합니다.

    다시 부르면 기록된 문항을 건너뜁니다. out_dir에 기록이 없고 drive_dir에 있으면 먼저 가져옵니다.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if drive_dir is not None:
        drive_dir = Path(drive_dir)
        drive_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    if not pred_path.is_file() and drive_dir is not None and (drive_dir / "predictions.jsonl").is_file():
        shutil.copy2(drive_dir / "predictions.jsonl", pred_path)
        print("Drive에서 이전 기록을 복구했습니다")
    done = set()
    if pred_path.is_file():
        done = {json.loads(l)["id"] for l in pred_path.open(encoding="utf-8") if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"[{desc or out_dir.name}] 대상 {len(rows)} / 기록됨 {len(done)} / 남은 {len(todo)}")

    letters = letter_token_ids(proc.tokenizer)
    instruction = ce.CONSENSUS_INSTRUCTION if consensus else ce.ANSWER_INSTRUCTION
    t_all = time.time()
    for start in range(0, len(todo), CHUNK):
        batch = todo[start:start + CHUNK]
        t0 = time.time()
        reqs, sizes = zip(*(make_request(proc, r, consensus, image_fn) for r in batch))
        if start == 0:
            assert "<|image_pad|>" in reqs[0]["prompt"], "프롬프트에 이미지 토큰이 없습니다"
            assert reqs[0]["prompt"].count(instruction) == 1, "지시문이 기대와 다릅니다"
        outs = llm.generate(list(reqs), sampling, use_tqdm=False)
        elapsed = time.time() - t0
        with pred_path.open("a", encoding="utf-8") as log:
            for row, size, o in zip(batch, sizes, outs):
                c = o.outputs[0]
                raw = c.text.strip()
                log.write(json.dumps(dict(
                    id=row["id"], raw_output=raw, pred=ce.parse_choice(raw),
                    letter_logprobs=letter_logprobs(c.logprobs[0], letters) if c.logprobs else {},
                    answer_truncated=c.finish_reason == "length", answer_tokens=len(c.token_ids),
                    seconds=elapsed / len(batch), image_width=size[0], image_height=size[1]),
                    ensure_ascii=False) + "\n")
        if drive_dir is not None:
            for f in out_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, drive_dir / f.name)
        n_done = start + len(batch)
        print(f"  {n_done}/{len(todo)}  {elapsed / len(batch):.3f}초/문항  "
              f"남은 약 {(time.time() - t_all) / n_done * (len(todo) - n_done) / 60:.0f}분")
    return read(pred_path)


def read(pred_path):
    res = pd.read_json(pred_path, lines=True).drop_duplicates("id", keep="last")
    res["conf"] = res.letter_logprobs.map(
        lambda lp: math.exp(max(lp.values())) if isinstance(lp, dict) and lp else float("nan"))
    return res


def fill_invalid(res):
    """제출용. 파서가 실패한 문항은 기록된 a~d 로그확률 중 가장 높은 글자로 채웁니다."""
    res = res.copy()
    bad = res.pred == "INVALID"
    res.loc[bad, "pred"] = res.loc[bad, "letter_logprobs"].map(
        lambda lp: max(lp, key=lp.get) if lp else "INVALID")
    return res, int(bad.sum())


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def paired(gold, pred_a, pred_b, name_a, name_b):
    """gold(id, answer)에 두 예측(id, pred)을 맞대 McNemar를 출력합니다."""
    j = (gold[["id", "answer"]]
         .merge(pred_a[["id", "pred"]].rename(columns={"pred": "a"}), on="id")
         .merge(pred_b[["id", "pred"]].rename(columns={"pred": "b"}), on="id"))
    ok_a, ok_b = j.a == j.answer, j.b == j.answer
    b, c = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    n = len(j)
    diff, se = (c - b) / n, math.sqrt(b + c) / n
    p = mcnemar_exact(b, c)
    print(f"[{n}문항] {name_a} {ok_a.mean():.4f} / {name_b} {ok_b.mean():.4f}")
    print(f"  {name_a}만 정답 {b} / {name_b}만 정답 {c} / 차이 {diff * 100:+.2f}pp "
          f"95%CI [{(diff - 1.96 * se) * 100:+.2f}, {(diff + 1.96 * se) * 100:+.2f}] / McNemar p = {p:.4g}")
    return dict(n=n, a_only=b, b_only=c, diff=diff, p=p)


def _self_test():
    assert abs(mcnemar_exact(11, 17) - 0.3449) < 1e-3
    res = pd.DataFrame({"id": ["x", "y"], "pred": ["INVALID", "b"],
                        "letter_logprobs": [{"a": -3.0, "b": -2.0, "c": -0.1, "d": -5.0}, {}]})
    filled, n = fill_invalid(res)
    assert n == 1 and list(filled.pred) == ["c", "b"]
    print("vllm_eval OK")


if __name__ == "__main__":
    _self_test()
