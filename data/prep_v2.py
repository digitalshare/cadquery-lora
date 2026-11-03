"""v2 data: same parts and prompts as prep.py, but CadQuery code rewritten into prompt units.

Original code uses DeepCAD integer units (longest sketch extent = 200) and a hidden centering
origin. We (1) execute the original program to get its bounding box, (2) compute K = longest
code extent / longest dimension stated in the expert prompt, (3) add the box center to every
origin (so part 1 sits at the prompt's translation vector), (4) divide every numeric literal by K,
and (5) re-execute the rewritten program and require the same box proportions.

python data/prep_v2.py 20000 1.5   -> data/train_v2.jsonl, data/val_v2.jsonl
"""
import glob, io, json, os, random, re, signal, sys, time, tokenize
from multiprocessing import Pool
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))
SYSTEM = open(os.path.join(ROOT, "data/system_prompt.txt")).read().strip()
N_UIDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
PROMPTS_PER_UID = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
N_VAL = 300
LEVELS = ["abstract", "beginner", "intermediate", "expert"]; LEVEL_W = [0.15, 0.25, 0.25, 0.35]
WORDS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}

def clean_code(code):
    code = code.replace("# Export\n", ""); code = re.sub(r"^\s*show_object\(.*\)\s*$", "", code, flags=re.M)
    return code.strip() + "\n"

def _alarm(signum, frame): raise TimeoutError()

def run_bbox(code):
    """Execute code; return (xmin,ymin,zmin,xmax,ymax,zmax) or None."""
    import cadquery as cq
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(8)
    try:
        ns = {"show_object": lambda *a, **k: None}; exec(compile(code, "<c>", "exec"), ns); r = ns.get("result")
        if r is None: return None
        vals = r.vals() if isinstance(r, cq.Workplane) else [r]
        shapes = [v for v in vals if isinstance(v, cq.Shape)]
        if not shapes or sum(s.Volume() for s in shapes) <= 1e-9: return None
        bb = cq.Compound.makeCompound(shapes).BoundingBox() if len(shapes) > 1 else shapes[0].BoundingBox()
        return (bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax)
    except Exception:
        return None
    finally:
        signal.alarm(0)

def words_to_num(s):
    toks = s.strip().split();
    if "point" not in toks: return None
    i = toks.index("point")
    try:
        ip = "".join(str(WORDS[t]) for t in toks[:i]); fp = "".join(str(WORDS[t]) for t in toks[i+1:])
        return float(f"{ip or 0}.{fp or 0}")
    except KeyError: return None

def stated_max_dim(expert):
    """Largest of the final length/width/height stated at the end of the expert prompt."""
    tail = expert[-700:]
    nums = [float(x) for x in re.findall(r"(?:length|width|height|long|wide|tall|high)[^0-9.]{0,25}(\d+\.\d+)", tail)]
    nums += [float(x) for x in re.findall(r"(\d+\.\d+)\s*units?\s*(?:in\s+)?(?:length|width|height|long|wide|tall|high)", tail)]
    if not nums:  # spelled-out numbers: "length zero point seven five units"
        for m in re.finditer(r"(?:length|width|height)\s+((?:(?:zero|one|two|three|four|five|six|seven|eight|nine|point)\s*)+)", tail):
            v = words_to_num(m.group(1))
            if v is not None: nums.append(v)
    nums = [n for n in nums if 0.01 <= n <= 1.5]
    return max(nums) if nums else None

def fmt(v):
    s = f"{v:.4f}".rstrip("0").rstrip("."); return s if s not in ("", "-0", "-") else "0"

def rewrite(code, bbox, K):
    cx, cy, cz = -bbox[0], -bbox[1], -bbox[2]  # move the box's min corner to the origin (the prompt's frame)
    def origin_sub(m):
        x, y, z = [float(t) for t in m.group(1).split(",")]
        return f"origin=({fmt((x+cx)/K)}, {fmt((y+cy)/K)}, {fmt((z+cz)/K)})"
    out_lines = []
    for line in code.splitlines():
        if "origin=" in line:
            out_lines.append(re.sub(r"origin=\(([^)]*)\)", origin_sub, line)); continue
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(line).readline))
        except tokenize.TokenError:
            out_lines.append(line); continue
        pieces, last = [], 0
        for t in toks:
            if t.type == tokenize.NUMBER:
                s, e = t.start[1], t.end[1]; pieces.append(line[last:s]); pieces.append(fmt(float(t.string)/K)); last = e
        pieces.append(line[last:]); out_lines.append("".join(pieces))
    return "\n".join(out_lines) + "\n"

def transform(args):
    uid, code, Lp = args
    if Lp is None: return uid, None, "nodim"
    bb = run_bbox(code)
    if bb is None: return uid, None, "exec0"
    ext = [bb[3]-bb[0], bb[4]-bb[1], bb[5]-bb[2]]; Lc = max(ext)
    if Lc <= 0: return uid, None, "flat"
    K = Lc / Lp
    new = rewrite(code, bb, K)
    bb2 = run_bbox(new)
    if bb2 is None: return uid, None, "exec1"
    ext2 = [bb2[3]-bb2[0], bb2[4]-bb2[1], bb2[5]-bb2[2]]
    if any(abs(a/K - b) > 0.02 * Lp + 1e-3 for a, b in zip(ext, ext2)): return uid, None, "mismatch"
    if min(bb2[:3]) < -0.05 or max(bb2[3:]) > 1.6: return uid, None, "offframe"  # min corner should sit at the origin like the prompt frame
    return uid, new, "ok"

def main():
    t0 = time.time()
    tr = pd.read_parquet(os.path.join(ROOT, "data/raw/cad-sft/text2cad-bench/train.parquet"))
    lv = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ROOT, "data/raw/t2c8k/data/*.parquet")))])
    lv["uid8"] = lv["uid"].str.split("/").str[-1]; lv = lv.drop_duplicates("uid8").set_index("uid8")
    rng = random.Random(0)
    tr = tr[tr["uid"].isin(lv.index)].sample(frac=1.0, random_state=0).head(int(N_UIDS * 1.15))  # same candidates as prep.py
    tr["code"] = tr["code"].map(clean_code)
    tr["Lp"] = [stated_max_dim(lv.loc[u, "expert"]) for u in tr["uid"]]
    print(f"candidates {len(tr)}; stated max dim parsed for {tr['Lp'].notna().mean():.1%}; median {tr['Lp'].median()}", flush=True)
    with Pool(8) as p:
        res = p.map(transform, list(zip(tr["uid"], tr["code"], tr["Lp"])), chunksize=32)
    from collections import Counter
    print("transform:", Counter(r[2] for r in res), f"in {time.time()-t0:.0f}s", flush=True)
    newcode = {u: c for u, c, s in res if c}
    ok = tr[tr["uid"].isin(newcode)].copy(); ok["code"] = ok["uid"].map(newcode)
    trn = ok.head(N_UIDS); va = ok.iloc[N_UIDS:N_UIDS + N_VAL]
    print("sample rewritten code:\n" + trn.iloc[0]["code"][:600], flush=True)

    def rows(df, k):
        out = []
        for uid, code in zip(df["uid"], df["code"]):
            row = lv.loc[uid]; n = int(k) + (1 if rng.random() < (k - int(k)) else 0)
            levels = rng.choices(LEVELS, LEVEL_W, k=4); seen = set()
            for L in levels:
                if len(seen) >= n: break
                desc = row[L]
                if not isinstance(desc, str) or len(desc) < 10 or L in seen: continue
                seen.add(L)
                out.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": desc.strip()},
                                         {"role": "assistant", "content": "```python\n" + code + "```"}], "level": L})
        rng.shuffle(out); return out
    for name, rs in [("train_v2", rows(trn, PROMPTS_PER_UID)), ("val_v2", rows(va, 1.0))]:
        with open(os.path.join(ROOT, f"data/{name}.jsonl"), "w") as f:
            for r in rs: f.write(json.dumps({"messages": r["messages"]}) + "\n")
        chars = sum(len(m["content"]) for r in rs for m in r["messages"])
        print(f"{name}: {len(rs)} examples, ~{chars/4/1e6:.1f}M tokens, levels={Counter(r['level'] for r in rs)}", flush=True)
    print(f"done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
