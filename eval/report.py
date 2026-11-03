import json, glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
VALID_ONLY = "--valid-gt" in sys.argv
FLAT = set(json.load(open(os.path.join(ROOT, "results/flat_gt_uids.json")))) if os.path.exists(os.path.join(ROOT, "results/flat_gt_uids.json")) else set()
import statistics
for p in sorted(glob.glob(os.path.join(ROOT, "results/*/summary.json"))):
    s = json.load(open(p)); d = os.path.dirname(p)
    recs = [json.loads(l) for l in open(os.path.join(d, "records.jsonl"))]
    if VALID_ONLY:  # drop parts whose benchmark ground-truth mesh is degenerate (flat), recompute exec/CD on the rest
        recs = [r for r in recs if r["uid"] not in FLAT]
        cds = [r["cd_mean"] for r in recs if r.get("ok") and r.get("cd_mean") is not None]
        s["exec_rate"] = sum(1 for r in recs if r.get("ok")) / len(recs); s["cd_median"] = statistics.median(cds) if cds else None
        s["cd_all_penalized"] = (sum(cds) + 100.0 * (len(recs) - len(cds))) / len(recs); s["n"] = len(recs)
    n = len(recs)
    usable = sum(1 for r in recs if r.get("ok") and r.get("cd_mean") is not None and r["cd_mean"] < 10) / n if n else 0
    s["usable_rate"] = usable  # executed AND Chamfer < 10 (x100): a part a human would accept as a starting point
    s["usable_per_dollar"] = (usable * 1000 / s["cost_per_1k_parts"]) if s.get("cost_per_1k_parts") else None
    rows.append(s)
cols = ["name", "level", "n", "exec_rate", "cd_median", "cd_all_penalized", "usable_rate", "iou_mean", "latency_p50", "cost_per_1k_parts", "usable_per_dollar"]
args = [a for a in sys.argv[1:] if not a.startswith("--")]; only = args[0] if args else None
print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
for r in rows:
    if r["name"].startswith("smoke"): continue
    if only and only not in r["name"]: continue
    f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) and abs(v) < 100 else (f"{v:.0f}" if isinstance(v, float) else str(v)))
    print("| " + " | ".join(f(r.get(c)) for c in cols) + " |")
