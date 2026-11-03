"""Build Fireworks SFT JSONL from Text2CAD multi-level prompts + CadQuery code.

Output: data/train.jsonl, data/val.jsonl  (OpenAI chat format)
"""
import glob, json, os, random, re, signal, sys, time
from multiprocessing import Pool
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM = open(os.path.join(ROOT, "data/system_prompt.txt")).read().strip()
N_UIDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
PROMPTS_PER_UID = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
N_VAL = 300
LEVELS = ["abstract", "beginner", "intermediate", "expert"]
LEVEL_W = [0.15, 0.25, 0.25, 0.35]  # eval (T1) uses L1=beginner-ish; keep expert heavy for precision

def clean_code(code):
    code = code.replace("# Export\n", "")
    code = re.sub(r"^\s*show_object\(.*\)\s*$", "", code, flags=re.M)
    return code.strip() + "\n"

def _alarm(signum, frame): raise TimeoutError()

def check(args):
    uid, code = args
    import cadquery as cq
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(8)
    try:
        ns = {"show_object": lambda *a, **k: None}
        exec(compile(code, "<c>", "exec"), ns)
        r = ns.get("result")
        if r is None: return uid, False
        if isinstance(r, cq.Workplane):
            vals = r.vals()
            if not vals: return uid, False
            vol = sum(v.Volume() for v in vals if isinstance(v, cq.Shape))
        else:
            vol = r.Volume()
        return uid, vol > 1e-9
    except Exception:
        return uid, False
    finally:
        signal.alarm(0)

def main():
    t0 = time.time()
    tr = pd.read_parquet(os.path.join(ROOT, "data/raw/cad-sft/text2cad-bench/train.parquet"))
    va = pd.read_parquet(os.path.join(ROOT, "data/raw/cad-sft/text2cad-bench/val.parquet"))
    lv = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ROOT, "data/raw/t2c8k/data/*.parquet")))])
    lv["uid8"] = lv["uid"].str.split("/").str[-1]
    lv = lv.drop_duplicates("uid8").set_index("uid8")
    print(f"train {len(tr)} val {len(va)} levels {len(lv)}; join coverage train={tr.uid.isin(lv.index).mean():.3f}")
    rng = random.Random(0)
    tr = tr[tr.uid.isin(lv.index)].sample(frac=1.0, random_state=0).head(int(N_UIDS * 1.15))
    va = va[va.uid.isin(lv.index)].sample(frac=1.0, random_state=0).head(int(N_VAL * 1.3))
    tr["code"] = tr.code.map(clean_code); va["code"] = va.code.map(clean_code)
    # verify execution
    with Pool(8) as p:
        res = dict(p.map(check, list(zip(tr["uid"], tr["code"])) + list(zip(va["uid"], va["code"])), chunksize=64))
    print(f"exec-verified: {sum(res.values())}/{len(res)} in {time.time()-t0:.0f}s")
    ok = tr[tr["uid"].map(res).fillna(False).astype(bool)]
    tr = ok.head(N_UIDS); va = ok.iloc[N_UIDS:N_UIDS + N_VAL]  # val = spare verified train-split parts (disjoint uids)
    print("cols", list(tr.columns), list(va.columns), len(tr), len(va))

    def rows(df, k):
        out = []
        for uid, code in zip(df["uid"], df["code"]):
            row = lv.loc[uid]
            n = int(k) + (1 if rng.random() < (k - int(k)) else 0)
            levels = rng.choices(LEVELS, LEVEL_W, k=4)
            seen = set()
            for L in levels:
                if len(seen) >= n: break
                desc = row[L]
                if not isinstance(desc, str) or len(desc) < 10 or L in seen: continue
                seen.add(L)
                out.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": desc.strip()},
                                         {"role": "assistant", "content": "```python\n" + code + "```"}], "uid": uid, "level": L})
        rng.shuffle(out); return out
    trows, vrows = rows(tr, PROMPTS_PER_UID), rows(va, 1.0)
    for name, rs in [("train", trows), ("val", vrows)]:
        with open(os.path.join(ROOT, f"data/{name}.jsonl"), "w") as f:
            for r in rs: f.write(json.dumps({"messages": r["messages"]}) + "\n")
        chars = sum(len(m["content"]) for r in rs for m in r["messages"])
        from collections import Counter
        print(f"{name}: {len(rs)} examples, ~{chars/4/1e6:.1f}M tokens, levels={Counter(r['level'] for r in rs)}")
    print(f"done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
