"""Build the static dashboard: dashboard/index.html (data + renders embedded) from existing eval results.

python dashboard/build.py            # writes dashboard/data.json, dashboard/renders/*.png, dashboard/index.html
"""
import base64, json, os, random, statistics, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "demo")); sys.path.insert(0, os.path.join(ROOT, "eval"))
from compare import render_png

R = os.path.join(ROOT, "results"); OUT = os.path.join(ROOT, "dashboard"); REN = os.path.join(OUT, "renders")
os.makedirs(REN, exist_ok=True)
GT = os.path.join(ROOT, "data/raw/textcad-bench/gt/gt_meshes")
T1 = {d["uid"].split("/")[-1]: d for d in json.load(open(os.path.join(ROOT, "data/raw/textcad-bench/t1_samples.json")))}

def load_run(name):
    d = os.path.join(R, name)
    if not os.path.exists(os.path.join(d, "summary.json")): return None
    s = json.load(open(os.path.join(d, "summary.json")))
    recs = {r["uid"]: r for r in map(json.loads, open(os.path.join(d, "records.jsonl")))}
    n = len(recs)
    s["usable_rate"] = sum(1 for r in recs.values() if r.get("ok") and r.get("cd_mean") is not None and r["cd_mean"] < 10) / n
    s["usable_per_dollar"] = s["usable_rate"] * 1000 / s["cost_per_1k_parts"] if s.get("cost_per_1k_parts") else None
    return s, recs

def metrics(s):
    return dict(exec=s["exec_rate"], usable=s["usable_rate"], cd_median=s["cd_median"], cd_pen=s["cd_all_penalized"],
                latency=s["latency_p50"], cost=s["cost_per_1k_parts"], upd=s["usable_per_dollar"], iou=s["iou_mean"])

MODELS = [  # key, label, kind, run prefix
    ("opus", "Claude Opus 5.5", "closed", "opus55"),
    ("haiku", "Claude Haiku 5.5", "closed", "haiku55"),
    ("deepseek", "DeepSeek V4.1 Flash", "open", "deepseek-v41-flash"),
    ("gptoss", "gpt-oss-120b", "open", "gpt-oss-120b"),
    ("base", "Qwen 3.6 35B-A3B untuned", "open", "qwen36-35b-base"),
    ("tuned", "Qwen 3.6 + LoRA (ours)", "ours", "tuned-v1-s100"),
]
runs = {}
models = []
for key, label, kind, pre in MODELS:
    m = dict(key=key, label=label, kind=kind)
    for lvl in ("l1", "l3"):
        x = load_run(f"{pre}-{lvl}")
        if x:
            runs[(key, lvl)] = x; m[lvl] = metrics(x[0])
    models.append(m)
# tuned latency measured alone
lat = load_run("latency-s100-l1")
if lat:
    for lvl in ("l1", "l3"):
        if lvl in models[-1]: models[-1][lvl]["latency"] = lat[0]["latency_p50"]; models[-1][lvl]["latency_note"] = "measured alone, 60 parts"

# checkpoints
state = json.load(open(os.path.join(R, "serverless/state-v1.json")))
ckpts = [dict(name="untuned", step=0)] + [dict(name=c["name"], step=c["step"]) for c in state["checkpoints"]]
for c in ckpts:
    pre = "qwen36-35b-base" if c["step"] == 0 else f"tuned-{c['name']}"
    for lvl in ("l1", "l3"):
        x = load_run(f"{pre}-{lvl}")
        if x: c[lvl] = metrics(x[0])
loss = [dict(step=s["step"], loss=round(s["loss"], 4)) for s in state["steps"]]

# gallery selection from L1 runs
base_s, base = runs[("base", "l1")]; tuned_s, tuned = runs[("tuned", "l1")]; opus_s, opus = runs[("opus", "l1")]
def cd(rs, u): r = rs.get(u, {}); return r.get("cd_mean") if r.get("ok") else None
cands = []
for u in tuned:
    cb, ct, co = cd(base, u), cd(tuned, u), cd(opus, u)
    if ct is None or co is None: continue
    if (cb is None or cb - ct > 15) and ct < 8: grp = "win"       # tuned much better than base, and good
    elif abs(ct - co) < 3 and ct < 10: grp = "parity"
    elif ct - co > 15: grp = "miss"
    else: continue
    cands.append((grp, u, cb, ct, co, T1[u]["n_operations"]))
random.Random(1).shuffle(cands)
pick = {"win": 6, "parity": 3, "miss": 3}; chosen = []
for grp, n in pick.items():
    pool = sorted([c for c in cands if c[0] == grp], key=lambda c: -c[5])  # prefer more complex parts
    chosen += pool[:n]
print({g: sum(1 for c in cands if c[0] == g) for g in pick}, "chosen", len(chosen))

def b64png(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
def render(uid, tag, src):
    png = os.path.join(REN, f"{uid}-{tag}.png")
    if not os.path.exists(png):
        try: render_png(src, png)
        except Exception as e: print("render fail", uid, tag, str(e)[:80]); return None
    return b64png(png)

gallery = []
for grp, u, cb, ct, co, nops in chosen:
    entry = dict(uid=u, group=grp, n_ops=nops, prompt=T1[u]["prompt_l1"], renders={}, cd={}, ok={}, code={}, latency={})
    entry["renders"]["gt"] = render(u, "gt", os.path.join(GT, u + ".obj"))
    for tag, rs, run in (("base", base, "qwen36-35b-base-l1"), ("tuned", tuned, "tuned-v1-s100-l1"), ("opus", opus, "opus55-l1")):
        r = rs.get(u, {}); entry["ok"][tag] = bool(r.get("ok")); entry["cd"][tag] = r.get("cd_mean"); entry["code"][tag] = r.get("code") or ""
        entry["latency"][tag] = r.get("latency"); entry["error"] = entry.get("error", {}); entry["error"][tag] = r.get("error") or ""
        stl = os.path.join(R, run, "stl", u + ".stl")
        entry["renders"][tag] = render(u, tag, stl) if r.get("ok") and os.path.exists(stl) else None
    gallery.append(entry)

tiles = dict(
    exec_before_l3=runs[("base", "l3")][0]["exec_rate"], exec_after_l3=runs[("tuned", "l3")][0]["exec_rate"], exec_opus_l3=runs[("opus", "l3")][0]["exec_rate"],
    exec_before_l1=base_s["exec_rate"], exec_after_l1=tuned_s["exec_rate"], exec_opus_l1=opus_s["exec_rate"],
    usable_before_l3=runs[("base", "l3")][0]["usable_rate"], usable_after_l3=runs[("tuned", "l3")][0]["usable_rate"], usable_opus_l3=runs[("opus", "l3")][0]["usable_rate"],
    usable_before_l1=base_s["usable_rate"], usable_after_l1=tuned_s["usable_rate"], usable_opus_l1=opus_s["usable_rate"],
    cost_after_l3=runs[("tuned", "l3")][0]["cost_per_1k_parts"], cost_opus_l3=runs[("opus", "l3")][0]["cost_per_1k_parts"],
    cost_after_l1=tuned_s["cost_per_1k_parts"], cost_opus_l1=opus_s["cost_per_1k_parts"],
    latency_after=lat[0]["latency_p50"] if lat else tuned_s["latency_p50"], latency_opus_l1=opus_s["latency_p50"], latency_opus_l3=runs[("opus", "l3")][0]["latency_p50"],
    train_tokens_m=3.11, train_cost=6.65, train_minutes=30, train_examples=6400, train_steps=100, loss_first=loss[0]["loss"], loss_last=loss[-1]["loss"],
    model_name="accounts/wjing63/models/cadq-qwen36-35b-v1", run_id=state["run_id"],
)
presets = [
    dict(label="Benchmark disc (tuned wins)", prompt=T1["00072791"]["prompt_l1"]),
    dict(label="Mounting bracket, free prose (tuned misses)", prompt="A rectangular mounting bracket: a flat base plate 0.75 long, 0.4 wide and 0.05 thick, with a vertical wall of the same length rising 0.3 at one long edge, and two through-holes of radius 0.04 in the base"),
    dict(label="Expert spec with coordinates", prompt=T1["00072791"]["prompt_l3"]),
]
data = dict(tiles=tiles, loss=loss, checkpoints=ckpts, models=models, gallery=gallery, presets=presets)
json.dump(data, open(os.path.join(OUT, "data.json"), "w"))
tpl = open(os.path.join(OUT, "template.html")).read()
html = tpl.replace("/*__DATA__*/null", json.dumps(data))
open(os.path.join(OUT, "index.html"), "w").write(html)
print("wrote dashboard/index.html", len(html) // 1024, "KB; gallery", len(gallery))
