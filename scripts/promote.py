"""Promote the latest (or named) sampler checkpoint of a serverless run to a deployable model.

python scripts/promote.py --tag v1 [--checkpoint v1-s150] --output-model-id cadq-qwen36-35b-v1
"""
import argparse, json, os
from datetime import datetime
from fireworks.training.sdk import FireworksClient
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--checkpoint", default=None)
ap.add_argument("--output-model-id", required=True); ap.add_argument("--delete-session", action="store_true"); a = ap.parse_args()
st = json.load(open(os.path.join(ROOT, f"results/serverless/state-{a.tag}.json")))
fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"])
rows = fw.list_training_session_checkpoints(st["session"])
cands = [r for r in rows if r.get("promotable") and (a.checkpoint is None or a.checkpoint in r.get("checkpointName", ""))]
print("promotable:", [(r["checkpointName"], r["createTime"]) for r in cands])
target = max(cands, key=lambda r: datetime.fromisoformat(r["createTime"].replace("Z", "+00:00")))
print("promoting", target["name"])
model = fw.promote_session_checkpoint(name=target["name"], output_model_id=a.output_model_id, base_model=st["base_model"])
print("promoted ->", model)
st.setdefault("promoted", []).append(dict(checkpoint=target["checkpointName"], model=str(model), output_model_id=a.output_model_id))
json.dump(st, open(os.path.join(ROOT, f"results/serverless/state-{a.tag}.json"), "w"), indent=2)
if a.delete_session:
    print("deleting session", st["session"], fw.delete_training_session(st["session"]))
