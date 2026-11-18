#!/usr/bin/env bash
# 실험 A 완료를 기다렸다가 stage1.jsonl을 백업하고 실험 D를 실행합니다.
set -u
cd "$(dirname "$0")/.."

RUN=output/paddleocr_vl16_halftrain/full
OCR=$RUN/ocr_llm
BACKUP=output/ocr_spotting_val672.jsonl
LOGDIR=output/logs
mkdir -p "$LOGDIR"

echo "[$(date +%H:%M:%S)] 실험 A 완료 대기 시작"
DEADLINE=$(( $(date +%s) + 4*3600 ))   # 최대 4시간
while [ ! -f "$OCR/metrics.json" ]; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[$(date +%H:%M:%S)] 시간 초과: A가 4시간 내에 끝나지 않았습니다." >&2
    exit 2
  fi
  S1=$( [ -f "$OCR/stage1.jsonl" ] && wc -l < "$OCR/stage1.jsonl" || echo 0 )
  S2=$( [ -f "$OCR/stage2.jsonl" ] && wc -l < "$OCR/stage2.jsonl" || echo 0 )
  echo "[$(date +%H:%M:%S)] 대기 중 — stage1 ${S1}/672, stage2 ${S2}/672"
  sleep 60
done

echo "[$(date +%H:%M:%S)] 실험 A 완료 감지"
LINES=$(wc -l < "$OCR/stage1.jsonl")
if [ "$LINES" -ne 672 ]; then
  echo "경고: stage1.jsonl 행 수가 672가 아닙니다 ($LINES). 백업은 진행합니다." >&2
fi
cp "$OCR/stage1.jsonl" "$BACKUP"
echo "[$(date +%H:%M:%S)] 백업 완료: $BACKUP ($LINES 행)"

# A의 커널이 GPU 메모리를 완전히 반환할 여유를 둡니다.
sleep 20

echo "[$(date +%H:%M:%S)] 실험 D 시작"
./baseline/Scripts/python.exe scripts/run_nb.py notebooks/experiments/AI_Challenge2_expD_llm_blind.ipynb 2>&1 | tee "$LOGDIR/exp_D.log"
STATUS=${PIPESTATUS[0]}
echo "[$(date +%H:%M:%S)] 실험 D 종료 코드: $STATUS"
exit "$STATUS"
