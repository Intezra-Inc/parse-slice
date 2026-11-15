#!/usr/bin/env bash
# Full OCR benchmark on one GPU, one engine at a time, nothing else of ours on that GPU:
#   crops (23,078 blocks) for the classic engines; full pages (1,651) for every engine, timed per page
#   (the 300-page speed figure = the first 300 pages of the same run, minus 5 warm-up pages).
# Engines: nemo-multi, nemo-en, ppocr (v5 server), ppocr6 (v6 medium), paddlevl (vLLM server on the same GPU),
#          parse35 (single), parse35d (double), parse35s (single + page slicing, release/parse-slice).
# Usage: run_full.sh <box label> <gpu> [vl port] [parse port]     e.g. run_full.sh h100 0   |   run_full.sh blackwell 1
# Output: preds_full/<box>/<engine>_<crops|pages>.jsonl   Log: logs/full_<box>.log   Ends with "FULL <box> DONE".
set -u
cd "$(dirname "$(readlink -f "$0")")"
BOX=$1; G=$2; VLP=${3:-8118}; PP=${4:-8119}; O=preds_full/$BOX; mkdir -p $O logs
LOG=logs/full_$BOX.log; say(){ echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
export CUDA_VISIBLE_DEVICES=$G PADDLE_PDX_CACHE_HOME=$PWD/cache-paddle
up(){ for i in $(seq 1 120); do curl -sf -m 3 localhost:$1/v1/models >/dev/null && return 0; sleep 5; done; say "server :$1 not up"; return 1; }
stop(){ pkill -f "$1" ; sleep 10; }
say "START box=$BOX gpu=$G ($(nvidia-smi --query-gpu=name --format=csv,noheader -i $G))"
for e in multi en; do
  for m in crops pages; do say "nemo-$e $m"; venv-nemo/bin/python -I run_nemo.py $m $e $O/nemo-${e}_$m.jsonl >> logs/full_${BOX}_nemo.log 2>&1; done
done
for e in ppocr ppocr6; do
  for m in crops pages; do say "$e $m"; venv-paddle/bin/python -I run_paddle.py $m $e $O/${e}_$m.jsonl >> logs/full_${BOX}_paddle.log 2>&1; done
done
say "paddlevl pages"
(GPU=$G PORT=$VLP UTIL=0.5 setsid nohup ./serve_vl.sh > logs/full_${BOX}_serve_vl.log 2>&1 < /dev/null &)
up $VLP && VL_URL=http://127.0.0.1:$VLP/v1 venv-paddle/bin/python -I run_paddle.py pages vl $O/paddlevl_pages.jsonl >> logs/full_${BOX}_paddle.log 2>&1
stop "served-model-name PaddleOCR-VL-1.6-0.9B .*--port $VLP"
(GPU=$G PORT=$PP UTIL=0.85 setsid nohup ./serve_parse.sh > logs/full_${BOX}_serve_parse.log 2>&1 < /dev/null &)
if up $PP; then
  export PARSE_URL=http://127.0.0.1:$PP/v1
  say "parse35 pages";  venv-vllm/bin/python -I run_parse.py $O/parse35_pages.jsonl "" single >> logs/full_${BOX}_parse.log 2>&1
  say "parse35d pages"; venv-vllm/bin/python -I run_parse.py $O/parse35d_pages.jsonl "" double >> logs/full_${BOX}_parse.log 2>&1
  say "parse35s pages"; venv-vllm/bin/python -I run_slice.py $O/parse35s_pages.jsonl >> logs/full_${BOX}_parse.log 2>&1
fi
stop "served-model-name nemotron-parse .*--port $PP"
for f in $O/*.jsonl; do say "$(basename $f): $(wc -l < $f) rows"; done
say "FULL $BOX DONE"
