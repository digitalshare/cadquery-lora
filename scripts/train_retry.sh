#!/usr/bin/env bash
# Relaunch serverless training until the session is created (pool outages return "no healthy upstream").
# usage: scripts/train_retry.sh <tag> <max-examples> [extra args...]
cd "$(dirname "$0")/.."
TAG=$1; N=$2; shift 2
export TINKER_TELEMETRY=0
for i in $(seq 1 40); do
  echo "[$(date +%H:%M:%S)] attempt $i" >> results/logs/train-$TAG.retries
  .venv/bin/python train/sft_serverless.py --max-examples "$N" --batch 64 --accum 1 --lr 1e-4 --warmup 8 --rank 16 --save-every 25 --tag "$TAG" "$@" > results/logs/train-$TAG.log 2>&1
  if grep -q "^done\." results/logs/train-$TAG.log; then echo "finished" >> results/logs/train-$TAG.retries; exit 0; fi
  if grep -q "^session=" results/logs/train-$TAG.log; then echo "ran but crashed after session creation; see log" >> results/logs/train-$TAG.retries; exit 1; fi
  sleep 60
done
