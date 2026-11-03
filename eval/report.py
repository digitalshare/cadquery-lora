import json, glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for p in sorted(glob.glob(os.path.join(ROOT, "results/*/summary.json"))):
    s = json.load(open(p)); d = os.path.dirname(p)
    recs = [json.loads(l) for l in open(os.path.join(d, "records.jsonl"))]
    n = len(recs)
    usable = sum(1 for r in recs if r.get("ok") and r.get("cd_mean") is not None and r["cd_mean"] < 10) / n if n else 0
    s["usable_rate"] = usable  # executed AND Chamfer < 10 (x100): a part a human would accept as a starting point
    s["usable_per_dollar"] = (usable * 1000 / s["cost_per_1k_parts"]) if s.get("cost_per_1k_parts") else None
    rows.append(s)
cols = ["name", "level", "n", "exec_rate", "cd_median", "cd_all_penalized", "usable_rate", "iou_mean", "latency_p50", "cost_per_1k_parts", "usable_per_dollar"]
only = sys.argv[1] if len(sys.argv) > 1 else None
print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
for r in rows:
    if r["name"].startswith("smoke"): continue
    if only and only not in r["name"]: continue
    f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) and abs(v) < 100 else (f"{v:.0f}" if isinstance(v, float) else str(v)))
    print("| " + " | ".join(f(r.get(c)) for c in cols) + " |")
