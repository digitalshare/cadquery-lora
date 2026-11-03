#!/usr/bin/env bash
# Score a serverless sampler checkpoint on T1 (abstract L1 + expert L3), 300 parts each, in parallel.
# usage: scripts/eval_ckpt.sh <tag> <ckpt-name e.g. v1-s25> [n]
cd "$(dirname "$0")/.."
TAG=$1; CK=$2; N=${3:-300}
PATH_=$(.venv/bin/python -c "import json,sys; s=json.load(open('results/serverless/state-$TAG.json')); print([c['path'] for c in s['checkpoints'] if c['name']=='$CK'][0])")
TOK=$(.venv/bin/python -c "import json; print(json.load(open('results/serverless/state-$TAG.json'))['tokenizer'])")
echo "checkpoint path: $PATH_"
export TINKER_TELEMETRY=0
.venv/bin/python eval/run.py --name "tuned-$CK-l1" --backend serverless --model "ckpt|$PATH_|$TOK" --level l1 --n "$N" --workers 6 --max-tokens 2048 > "results/logs/tuned-$CK-l1.log" 2>&1 &
.venv/bin/python eval/run.py --name "tuned-$CK-l3" --backend serverless --model "ckpt|$PATH_|$TOK" --level l3 --n "$N" --workers 6 --max-tokens 2048 > "results/logs/tuned-$CK-l3.log" 2>&1 &
wait
.venv/bin/python eval/report.py | grep -E "name|---|tuned-$CK"
