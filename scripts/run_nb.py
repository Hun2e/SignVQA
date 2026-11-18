"""노트북의 코드 셀을 순서대로 실행합니다 (nbconvert 없이).

사용: python scripts/run_nb.py <notebook.ipynb> [--from "<마크다운 제목 일부>"]

--from을 주면 첫 코드 셀(설정)을 실행한 뒤, 그 글자가 들어 있는 마크다운 셀 이후의 코드 셀만 실행합니다.
예: 검증은 이미 끝났고 제출 추론만 돌릴 때 `--from "제출 추론"`.

노트북을 단일 원본으로 두고 그대로 실행하므로, 스크립트 사본이 따로 갈라지지 않습니다.
셀 하나가 실패하면 그 셀 번호와 소스를 함께 출력하고 중단합니다.
"""

from _project import ROOT
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # 창 없는 환경에서 plt.show()가 멈추지 않도록


def main():
    args = sys.argv[1:]
    start_marker = None
    if len(args) == 3 and args[1] == "--from":
        start_marker = args[2]
    elif len(args) != 1:
        raise SystemExit('사용: python scripts/run_nb.py <notebook.ipynb> [--from "<제목 일부>"]')
    path = Path(args[0]).resolve()
    os.chdir(ROOT)
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
             if c["cell_type"] == "code" and "".join(c["source"]).strip()]
    if start_marker is not None:
        marks = [i for i, c in enumerate(nb["cells"])
                 if c["cell_type"] == "markdown" and start_marker in "".join(c["source"])]
        if not marks:
            raise SystemExit(f"'{start_marker}'가 들어 있는 마크다운 셀이 없습니다")
        cells = [cells[0]] + [(i, s) for i, s in cells[1:] if i > marks[0]]
    print(f"=== {path.name}: 코드 셀 {len(cells)}개 실행 ===", flush=True)
    env = {"__name__": "__main__", "__file__": str(path)}
    for n, (index, src) in enumerate(cells, 1):
        print(f"\n--- [{n}/{len(cells)}] cell {index} ---", flush=True)
        try:
            exec(compile(src, f"{path.name}:cell{index}", "exec"), env)
        except Exception:
            print(f"\n!!! cell {index} 실패 !!!", flush=True)
            print(src, flush=True)
            traceback.print_exc()
            raise SystemExit(1)
    print(f"\n=== {path.name} 완료 ===", flush=True)


if __name__ == "__main__":
    main()
