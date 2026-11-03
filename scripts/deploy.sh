#!/usr/bin/env bash
# Deploy a model (base or LoRA via live-merge) to an on-demand deployment and wait until READY.
# usage: scripts/deploy.sh <model-resource-name> <deployment-id> [shape]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:?model}; DEP=${2:?deployment id}; SHAPE=${3:-default}
F="./bin/firectl --api-key $FIREWORKS_API_KEY"
$F deployment create "$MODEL" --deployment-id "$DEP" --deployment-shape "$SHAPE" --min-replica-count 1 --max-replica-count 1 -o json | tee "results/deploy-$DEP.json"
for i in $(seq 1 90); do
  S=$($F deployment get "accounts/wjing63/deployments/$DEP" -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('state'))")
  echo "$(date +%H:%M:%S) $DEP state=$S"
  [ "$S" = "READY" ] && break
  sleep 20
done
