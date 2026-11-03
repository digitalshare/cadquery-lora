#!/usr/bin/env bash
# Upload datasets and launch managed LoRA SFT on Fireworks.
# usage: scripts/train.sh <run-tag> [base-model-id] [epochs] [lora-rank]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?run tag}; BASE=${2:-qwen3p5-9b}; EPOCHS=${3:-1}; RANK=${4:-16}
F="./bin/firectl --api-key $FIREWORKS_API_KEY"
DS_TRAIN="cadq-train-$TAG"; DS_VAL="cadq-val-$TAG"; OUT="cadq-$BASE-$TAG"
$F dataset create "$DS_TRAIN" data/train.jsonl
$F dataset create "$DS_VAL" data/val.jsonl
$F sftj create \
  --base-model "accounts/fireworks/models/$BASE" \
  --dataset "$DS_TRAIN" --evaluation-dataset "$DS_VAL" \
  --output-model "$OUT" --display-name "text2cadquery-$TAG" \
  --epochs "$EPOCHS" --lora-rank "$RANK" \
  --learning-rate-scheduler cosine --learning-rate-min-lr-ratio 0.1 \
  --max-context-length 4096 --batch-size-samples 32 \
  -o json | tee "results/sftj-$TAG.json"
echo "output model: accounts/wjing63/models/$OUT"
