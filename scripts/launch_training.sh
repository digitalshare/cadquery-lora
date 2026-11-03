#!/usr/bin/env bash
# One-shot: upload datasets + create the SFT job. Run after the Fireworks account has dataset uploads enabled.
set -euo pipefail; cd "$(dirname "$0")/.."
P=.venv/bin/python
$P scripts/fw_api.py dataset-create cadq-train-v1 data/train.jsonl
$P scripts/fw_api.py dataset-create cadq-val-v1 data/val.jsonl
$P scripts/fw_api.py sftj-create results/sftj-spec-v1.json cadq-v1 | tee results/sftj-v1-create.json
echo "poll with: $P scripts/fw_api.py sftj-get cadq-v1"
