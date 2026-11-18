"""E6 이후 학습·평가 분할을 만들어 splits/e6_split.csv로 저장합니다.

E1~E4는 학습이 없어 val 672만으로 비교했습니다. E6부터 QLoRA를 하므로 학습에 쓰지 않을 평가 분할이
필요하고, val 672로는 1%p 안팎의 차이를 가리지 못했으므로(E4 val 96.88% vs LB 0.95710) 키웁니다.

- holdout 1,300 = 기존 val 672 전부 + 나머지에서 정답 비율을 맞춰 뽑은 628
- train 5,414 = 그 밖의 전부

val 672를 holdout에 통째로 넣어 지금까지의 기록과 이어지게 합니다. 분할은 sklearn 없이 pandas로
seed 42 고정 추출하고, 결과 파일과 해시를 저장소에 남겨 다시 추첨하지 않습니다.
"""

from _project import ROOT
import hashlib
import sys

import pandas as pd

sys.path.insert(0, str(ROOT / "src"))
import common_eval as ce  # noqa: E402

HOLDOUT_SIZE = 1300
SEED = 42
OUT = ROOT / "splits" / "e6_split.csv"

ce.DATA_DIR = ROOT / "data"
val = ce.load_val(ROOT / ce.BASELINE_RUN)
train = pd.read_csv(ROOT / "data/train.csv", keep_default_na=False)
rest = train[~train.id.isin(set(val.id))]
need = HOLDOUT_SIZE - len(val)

# 정답 비율을 맞춰 뽑습니다. 반올림으로 생긴 차이는 가장 큰 몫에 몰아서 보정합니다.
share = rest.answer.value_counts(normalize=True).sort_index()
quota = (share * need).round().astype(int)
quota[quota.idxmax()] += need - quota.sum()
extra = pd.concat([rest[rest.answer == k].sample(n=int(n), random_state=SEED)
                   for k, n in quota.items()])

holdout_ids = set(val.id) | set(extra.id)
split = train[["id", "answer"]].copy()
split["split"] = split.id.map(lambda i: "holdout" if i in holdout_ids else "train")
split["in_val672"] = split.id.isin(set(val.id))
assert (split.split == "holdout").sum() == HOLDOUT_SIZE
assert split[split.in_val672].split.eq("holdout").all()

OUT.parent.mkdir(exist_ok=True)
split[["id", "split", "in_val672"]].to_csv(OUT, index=False)
digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
print(split.groupby("split").answer.value_counts().unstack())
print(f"저장: {OUT} / sha256 {digest}")
