"""Verify the generated E2 (CoT) Colab notebook without a GPU.

E2는 test.csv 6714건에 4~5시간이 걸립니다. 폴백 분기 하나가 잘못되면 그 시간을 통째로
버리므로, GPU 없이 확인할 수 있는 것은 여기서 다 확인합니다.

1. E1과 공유하는 배관(적재·생성·로짓·파일명)이 글자 단위로 같은가
2. CoT 프롬프트가 E1과 마지막 지시문 한 줄만 다른가
3. 폴백 4분기(정상 / INVALID→재질의 / INVALID→로짓 / 잘림→폴백)가 의도대로 도는가
4. predictions.jsonl 스키마와 pred가 항상 a~d인가
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ast
import copy
import json
import tempfile
from types import ModuleType, SimpleNamespace

# common_eval은 모듈 최상단에서 pandas를 import하지만, 이 파일이 쓰는 함수는 쓰지 않습니다.
try:
    import pandas  # noqa: F401
except ImportError:
    sys.modules["pandas"] = ModuleType("pandas")

import common_eval as ce

NB_DIR = ROOT / "notebooks/experiments"
E1 = json.loads((NB_DIR / "AI_Challenge2_expE1_qwen35_9b_colab.ipynb").read_text(encoding="utf-8"))
E2 = json.loads((NB_DIR / "AI_Challenge2_expE2_qwen35_9b_cot_colab.ipynb").read_text(encoding="utf-8"))


def parse_code(text):
    """IPython shell escapes are parsed separately by IPython, not Python ast."""
    return ast.parse("\n".join(line[:len(line) - len(line.lstrip())] + "pass"
                               if line.lstrip().startswith("!") else line
                               for line in text.splitlines()))


def functions(nb):
    result = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in parse_code("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef):
                result[node.name] = node
    return result


def strip_doc(node):
    node = copy.deepcopy(node)
    first = node.body[0] if node.body else None
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        node.body = node.body[1:]
    return node


def code_text(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


f1, f2 = functions(E1), functions(E2)

# --- 1. E1과 공유하는 배관은 손대지 않았어야 합니다 ---------------------------
for name in ["load_vlm", "generate_text", "letter_token_ids", "submission_filename"]:
    assert ast.dump(f1[name]) == ast.dump(f2[name]), f"{name}이 E1과 다릅니다"
# choose_by_logit은 docstring만 바꿨습니다 (CoT 입력에 쓰면 안 된다는 경고).
assert ast.dump(strip_doc(f1["choose_by_logit"])) == ast.dump(strip_doc(f2["choose_by_logit"]))
assert "build_messages_plain" in f2 and "build_messages_plain" not in f1
assert "answer_without_cot" in f2 and "answer_without_cot" not in f1
print("PASS: 공유 배관 4개 동일 / choose_by_logit은 docstring만 변경 / 폴백 2개 추가")

# --- 2. 설정값과 RESUME 가드 -------------------------------------------------
t1, t2 = code_text(E1), code_text(E2)
for needle in ['MAX_NEW_TOKENS = 512', 'RETRY_MAX_NEW_TOKENS = 24', 'RETRY_ON_TRUNCATED = True',
               'SUBMISSION_METHOD  = "cot"', 'exp_E2_{STEM}', '"cot": True',
               'ce.parse_choice_cot(raw)', 'ce.SYSTEM_COT', 'cot=True',
               'SUBMISSION_CSV     = DATA_ROOT / "test.csv"']:
    assert needle in t2, f"E2에 없습니다: {needle}"
# 기본 대상은 E1과 같은 test.csv입니다. 저장된 E1 예측을 그대로 맞댈 수 있어야 합니다.
assert 'SUBMISSION_CSV     = DATA_ROOT / "test.csv"' in t1, "E1 기본값이 바뀌었습니다"
assert "MAX_NEW_TOKENS = 24" in t1 and "exp_E1_{STEM}" in t1
# RUN_META가 다르면 E1 예측이 남은 폴더에 E2를 이어 붙이려 할 때 RESUME이 막아 줍니다.
assert '"cot"' not in t1, "E1 RUN_META에 cot가 생겼다면 가드 근거를 다시 확인하세요"
config = "".join(E2["cells"][19]["source"])           # 출력 경로와 RUN_META를 정하는 셀
assert "exp_E1_" not in config and "zeroshot" not in config, "설정 셀에 E1 흔적이 남아 있습니다"
# exp_E1_은 짝 비교 셀에서 E1 예측을 읽을 때만 나와야 합니다.
owners = [i for i, c in enumerate(E2["cells"])
          if c["cell_type"] == "code" and "exp_E1_" in "".join(c["source"])]
assert owners == [35], f"exp_E1_ 참조가 짝 비교 셀 밖에 있습니다: {owners}"
print("PASS: CoT 설정값 / 출력 폴더 분리 / RESUME 가드(cot 키)")

# --- 3. 프롬프트는 마지막 지시문 한 줄만 달라야 합니다 ------------------------
scope = {"ce": ce}
for name in ["build_messages", "build_messages_plain"]:
    exec(compile(ast.Module(body=[f2[name]], type_ignores=[]), "<prompt>", "exec"), scope)

ROW = {"question": "이 간판의 상호명은?", "a": "가게1", "b": "가게2", "c": "가게3", "d": "가게4"}
cot_msg = scope["build_messages"](ROW, None)
plain_msg = scope["build_messages_plain"](ROW, None)
assert cot_msg[0]["content"][0]["text"] == ce.SYSTEM_COT
assert plain_msg[0]["content"][0]["text"] == ce.SYSTEM
cot_text = cot_msg[1]["content"][1]["text"]
plain_text = plain_msg[1]["content"][1]["text"]
assert cot_text.rsplit("\n", 1)[0] == plain_text.rsplit("\n", 1)[0], "질문/선지 본문이 다릅니다"
assert cot_text.endswith(ce.COT_INSTRUCTION) and plain_text.endswith(ce.ANSWER_INSTRUCTION)
# 폴백 재질의는 E1과 글자 단위로 같은 프롬프트여야 합니다.
e1_build = {"ce": ce}
exec(compile(ast.Module(body=[f1["build_messages"]], type_ignores=[]), "<e1>", "exec"), e1_build)
assert e1_build["build_messages"](ROW, None) == plain_msg, "재질의가 E1 프롬프트와 다릅니다"
print("PASS: CoT 프롬프트는 마지막 한 줄만 차이 / 재질의는 E1과 동일")

# --- 4. 폴백 4분기 ------------------------------------------------------------
# CoT 출력 -> (원문, 잘림) / 재질의 출력. 네 가지 경로를 모두 태웁니다.
SCRIPT = {
    "ok":         (("간판을 보면 ...\n정답: b", False), None,     "b", False, ""),
    "invalid":    (("설명만 하고 끝났다", False),      "c",       "c", True,  "retry"),
    "no_parse":   (("설명만 하고 끝났다", False),      "역시 모름", "d", True,  "logit"),
    "truncated":  (("...\n정답: a", True),             "b",       "b", True,  "retry"),
}


class Proc:
    """system 텍스트로 CoT 호출과 재질의 호출을 구분합니다."""

    def apply_chat_template(self, messages, **kwargs):
        system = messages[0]["content"][0]["text"]
        return {"mode": "cot" if system == ce.SYSTEM_COT else "plain"}


class CeShim:
    """open_image만 가짜로 바꾸고 나머지는 진짜 common_eval을 씁니다."""

    def __getattr__(self, name):
        return getattr(ce, name)

    def open_image(self, row):
        return SimpleNamespace(width=960, height=573)


current = {"id": None}


def fake_generate_text(model, proc, inputs, limit):
    cot_out, retry_out, *_ = SCRIPT[current["id"]]
    if inputs["mode"] == "cot":
        assert limit == 512, "CoT 호출이 512토큰이 아닙니다"
        raw, capped = cot_out
        return raw, capped, len(raw)
    assert limit == 24, "재질의가 24토큰이 아닙니다"
    return retry_out, False, len(retry_out or "")


def fake_choose_by_logit(model, proc, inputs, table):
    assert inputs["mode"] == "plain", "로짓 폴백이 CoT 입력 위에서 불렸습니다"
    return "d", {}


tmp = Path(tempfile.mkdtemp())
scope.update({
    "torch": SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None,
                                                  max_memory_allocated=lambda: 0)),
    "time": SimpleNamespace(perf_counter=lambda: 0.0),
    "json": json, "shutil": SimpleNamespace(copy2=lambda *a: None),
    "tqdm": lambda it, **kw: it,
    "ce": CeShim(), "model": None, "proc": Proc(),
    "generate_text": fake_generate_text, "choose_by_logit": fake_choose_by_logit,
    "ENABLE_THINKING": False, "RETRY_MAX_NEW_TOKENS": 24, "MAX_NEW_TOKENS": 512,
    "RETRY_ON_TRUNCATED": True, "SYNC_EVERY": 1000,
    "LETTERS": {ch: [i] for i, ch in enumerate(ce.CHOICES)},
    "PRED_PATH": tmp / "predictions.jsonl", "DRIVE_PRED": tmp / "drive.jsonl",
    "META_PATH": tmp / "run_meta.json", "DRIVE_DIR": tmp,
})
for name in ["answer_without_cot", "run_items"]:
    exec(compile(ast.Module(body=[f2[name]], type_ignores=[]), "<run>", "exec"), scope)


class Items(list):
    """run_items가 한 건씩 돌 때 어떤 시나리오인지 알려 줍니다."""

    def __iter__(self):
        for row in list.__iter__(self):
            current["id"] = row["id"]
            yield row


rows = Items({**ROW, "id": key, "path": f"{key}.jpg"} for key in SCRIPT)
scope["run_items"](rows, "verify")

records = [json.loads(line) for line in (tmp / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(records) == len(SCRIPT), f"기록 {len(records)}건 != 대상 {len(SCRIPT)}건"
FIELDS = {"id", "raw_output", "pred", "fallback", "fallback_stage", "retry_raw_output",
          "answer_truncated", "answer_tokens", "seconds", "image_width", "image_height"}
for rec in records:
    _, _, want_pred, want_fb, want_stage = SCRIPT[rec["id"]]
    assert set(rec) == FIELDS, f"{rec['id']} 스키마 불일치: {set(rec) ^ FIELDS}"
    assert rec["pred"] == want_pred, f"{rec['id']}: pred {rec['pred']} != {want_pred}"
    assert rec["fallback"] is want_fb, f"{rec['id']}: fallback {rec['fallback']}"
    assert rec["fallback_stage"] == want_stage, f"{rec['id']}: stage {rec['fallback_stage']}"
    assert rec["pred"] in ce.CHOICES, f"{rec['id']}: a~d가 아닙니다"
print("PASS: 폴백 4분기 (정상 / INVALID→재질의 / INVALID→로짓 / 잘림→폴백)")
print("PASS: predictions.jsonl 스키마 11필드, pred는 항상 a~d, INVALID 0건")

# --- 5. 짝 비교 통계 (16번 섹션) ----------------------------------------------
# 로컬 4bit 검증 672문항의 실제 분할표를 그대로 재현해, 노트북이 같은 결론을 내는지 봅니다.
import io
import math
from contextlib import redirect_stdout
from math import comb

stats = {"math": math, "comb": comb, "Path": Path, "pd": None, "E1_KAGGLE": 0.94727}
for name in ["wilson", "mcnemar_exact", "paired_report", "kaggle_delta"]:
    exec(compile(ast.Module(body=[f2[name]], type_ignores=[]), "<stats>", "exec"), stats)

assert abs(stats["mcnemar_exact"](11, 17) - 0.3449) < 5e-4, "McNemar 값이 어긋납니다"
assert stats["mcnemar_exact"](0, 0) == 1.0 and stats["mcnemar_exact"](9, 9) == 1.0
assert stats["mcnemar_exact"](110, 170) < 0.001, "10배 표본에서는 유의해야 합니다"
EPS = 1e-9   # k=0·k=n에서 Wilson 상·하한은 부동소수점 오차만큼 0/1을 스칩니다.
for k, n in [(633, 672), (0, 100), (100, 100)]:
    p, lo, hi = stats["wilson"](k, n)
    assert -EPS <= lo <= p + EPS and p - EPS <= hi <= 1 + EPS, \
        f"Wilson 구간이 [0,1]을 벗어납니다: {lo}, {hi}"


class Series(dict):
    """paired_report가 쓰는 pandas Series의 최소 동작만 흉내냅니다."""

    @property
    def index(self):
        return self.keys()


gold, e1_pred, e2_pred = Series(), Series(), Series()
for n, (p1, p2) in enumerate([("a", "a")] * 622 + [("b", "b")] * 22
                             + [("a", "b")] * 11 + [("b", "a")] * 17):
    key = f"train_{n:04d}.jpg"
    gold[key], e1_pred[key], e2_pred[key] = "a", p1, p2

buffer = io.StringIO()
with redirect_stdout(buffer):
    stats["paired_report"]("검증 재현", list(gold), e1_pred, e2_pred, gold)
report = buffer.getvalue()
for needle in ["672문항", "0.9420 (633/672)", "0.9509 (639/672)",
               "둘 다 정답 622 / 둘 다 오답 22 / E1만 11 / E2만 17",
               "+0.89pp", "[-0.65pp, +2.44pp]", "p = 0.3449", "구분되지 않습니다",
               "0.9673"]:
    assert needle in report, f"짝 비교 출력에 없습니다: {needle}\n{report}"
print("PASS: 짝 비교가 로컬 672 결과를 재현 (E1 0.9420 / E2 0.9509 / p = 0.3449 / 상한 0.9673)")

# 정답이 없을 때(test.csv)의 보수적 판정. 불일치 m 전부가 정오를 가른다고 놓은 상한입니다.
N_TEST, M_TEST = 6714, 330


def delta_text(new_score, **kw):
    out = io.StringIO()
    with redirect_stdout(out):
        stats["kaggle_delta"](new_score, n_scored=N_TEST, m=M_TEST, **kw)
    return out.getvalue()


small = delta_text(0.94727 + 0.004)        # +0.4%p ~= 27문항
assert "유의하다고 말할 수 없습니다" in small, small
big = delta_text(0.94727 + 0.030)          # +3.0%p ~= 201문항
assert "이 가정에서도 유의합니다" in big, big
# 점수 차이가 불일치 문항보다 크면 n_scored가 LB 채점 행 수와 어긋난 것입니다.
assert "경고" in delta_text(0.94727 + 0.20), "불가능한 점수 차를 잡지 못했습니다"
# 보수적 판정은 정답을 아는 검정보다 결코 느슨하면 안 됩니다.
loose = stats["mcnemar_exact"]((M_TEST - 27) // 2, M_TEST - (M_TEST - 27) // 2)
assert loose >= stats["mcnemar_exact"](11, 17) or loose > 0.05
print("PASS: 정답 없는 보수적 McNemar (작은 차이는 기각, 큰 차이는 유의, 불가능한 차이는 경고)")

# --- 6. 노트북 자체 ----------------------------------------------------------
ce._self_test()
try:
    import nbformat
except ImportError:
    print("nbformat 없음; JSON과 Python AST만 검증했습니다")
else:
    nbformat.validate(E2)
    print("PASS: nbformat 검증")
print(f"PASS: 셀 {len(E2['cells'])}개 / GPU 추론과 Drive 동기화는 실행하지 않았습니다")
