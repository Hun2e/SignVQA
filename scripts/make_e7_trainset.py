"""E7 QLoRA 학습 구성 목록을 splits/e7_train_<구성>.csv로 만듭니다.

사용: python scripts/make_e7_trainset.py A|B|C

- A: E6 학습 분할 5,414 전부
- B: A에서 splits/e7_exclude.csv에 적은 문항을 뺀 것. E6에서 모델이 확신하는데 사람과 어긋난 문항을
     이미지로 확인해(splits/e7_review_highconf.csv) 선명한 글자와 정답이 어긋나거나 전제가 틀린 것만 적습니다.
     사람다운 해석이나 모델 오답은 빼지 않습니다(모델이 틀린 문항 전부가 아님)
- C: A + dev에서 5명 중 3표 이상 단독 최다 답이 있는 문항

목록에는 id와 source만 담습니다. 정답 글자는 저장소에 두지 않고 Colab에서 train.csv·dev.csv로 붙입니다
(src/qlora.py rows_from_manifest). holdout 1,300은 어떤 구성에도 들어가지 않습니다.
"""

from _project import ROOT
import sys

import pandas as pd

sys.path.insert(0, str(ROOT / "src"))
import qlora  # noqa: E402

config = sys.argv[1] if len(sys.argv) == 2 else None
if config not in {"A", "B", "C"}:
    raise SystemExit("사용: python scripts/make_e7_trainset.py A|B|C")

split = pd.read_csv(ROOT / "splits/e6_split.csv")
base = split[split.split == "train"][["id"]].assign(source="train")

if config == "A":
    out = base
elif config == "B":
    path = ROOT / "splits/e7_exclude.csv"
    if not path.is_file():
        raise SystemExit(f"{path}가 없습니다. E6 불일치 문항을 분석해 뺄 문항(id, reason)을 먼저 정하세요")
    exclude = pd.read_csv(path)
    unknown = set(exclude.id) - set(base.id)
    assert not unknown, f"학습 분할에 없는 id: {sorted(unknown)[:3]}"
    out = base[~base.id.isin(set(exclude.id))]
    print(f"제외 {len(exclude)}문항. 사유별:", exclude.reason.value_counts().to_dict())
else:
    dev = pd.read_csv(ROOT / "data/dev.csv", keep_default_na=False)
    keep = dev[dev.apply(lambda r: qlora.dev_majority(r) is not None, axis=1)]
    out = pd.concat([base, keep[["id"]].assign(source="dev")], ignore_index=True)
    print(f"dev {len(dev)}문항 중 3표 이상 단독 최다 {len(keep)}문항 추가")

holdout = set(split[split.split == "holdout"].id)
assert not set(out.id) & holdout
dest = ROOT / "splits" / f"e7_train_{config}.csv"
out.to_csv(dest, index=False)
print(f"구성 {config}: {len(out)}문항 ({out.source.value_counts().to_dict()}) -> {dest}")
