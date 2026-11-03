"""Text-to-CadQuery evaluation runner.

python eval/run.py --name claude-opus --backend anthropic --model claude-opus-5-5 --level l1 --n 300
python eval/run.py --name qwen3-8b-base --backend fireworks --model accounts/.../models/qwen3-8b#... --level l1 --n 300
"""
import argparse, json, os, random, re, sys, time, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))
from cadexec import run_cadquery
from metrics import evaluate_pair

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T1 = os.path.join(ROOT, "data/raw/textcad-bench/t1_samples.json")
GT = os.path.join(ROOT, "data/raw/textcad-bench/gt/gt_meshes")

SYSTEM = open(os.path.join(ROOT, "data/system_prompt.txt")).read().strip()

# $/M tokens (input, output). Update from pricing page.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.1, 0.5),
    "gpt-5.5": (1.25, 10.0),
    "accounts/fireworks/models/deepseek-v4p1-flash-1017": (0.3, 1.2),
    "accounts/fireworks/models/gpt-oss-120b": (0.15, 0.6),
    "base|accounts/fireworks/models/qwen3p6-35b-a3b|Qwen/Qwen3.6-35B-A3B": (0.45, 1.335),
}

def extract_code(text):
    if "</think>" in text: text = text.split("</think>")[-1]
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = max(m, key=len) if m else re.sub(r"^\s*```(?:python)?\s*\n", "", text)
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.S).strip()
    if "</think>" in code: code = code.split("</think>")[-1].strip()
    return code

def gen_fireworks(model, user, max_tokens, temperature=0.0):
    from openai import OpenAI
    cli = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"], timeout=180, max_retries=3)
    t = time.time()
    r = cli.chat.completions.create(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                    max_tokens=max_tokens, temperature=temperature)
    dt = time.time() - t
    return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens, dt

def gen_anthropic(model, user, max_tokens, temperature=0.0):
    import anthropic
    cli = anthropic.Anthropic(timeout=180, max_retries=3)
    t = time.time()
    kw = dict(model=model, system=SYSTEM, messages=[{"role": "user", "content": user}], max_tokens=max_tokens)
    if os.environ.get("ANTHROPIC_EFFORT"):
        kw["output_config"] = {"effort": os.environ["ANTHROPIC_EFFORT"]}
    r = cli.messages.create(**kw)
    dt = time.time() - t
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens, dt

def gen_openai(model, user, max_tokens, temperature=0.0):
    from openai import OpenAI
    cli = OpenAI(timeout=180, max_retries=3)
    t = time.time()
    kw = dict(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], max_completion_tokens=max_tokens)
    r = cli.chat.completions.create(**kw)
    dt = time.time() - t
    return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens, dt

_SAMPLERS = {}
import threading
_SAMPLER_LOCK = threading.Lock()
def _sampler(spec):
    """spec: 'base|<base_model>|<hf_tokenizer>' or 'ckpt|<checkpoint_path>|<hf_tokenizer>'"""
    with _SAMPLER_LOCK:
        if spec not in _SAMPLERS:
            from fireworks.training.sdk import FiretitanServiceClient
            from transformers import AutoTokenizer
            kind, ref, tok_id = spec.split("|", 2)
            tok = AutoTokenizer.from_pretrained(tok_id)
            svc = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/training/v1/serverless")
            smp = svc.create_sampling_client(base_model=ref, tokenizer=tok) if kind == "base" else svc.create_sampling_client(model_path=ref, tokenizer=tok)
            _SAMPLERS[spec] = (svc, smp, tok)
        return _SAMPLERS[spec]

def gen_serverless(model, user, max_tokens, temperature=0.0):
    from fireworks.training.sdk import types
    svc, smp, tok = _sampler(model)
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t = time.time()
    r = smp.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1,
                   sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=max(temperature, 1e-6), stop=[tok.eos_token_id])).result()
    dt = time.time() - t
    seq = r.sequences[0]; out_ids = [int(x) for x in seq.tokens]
    text = tok.decode(out_ids, skip_special_tokens=True)
    return text, len(ids), len(out_ids), dt

BACKENDS = {"fireworks": gen_fireworks, "anthropic": gen_anthropic, "openai": gen_openai, "serverless": gen_serverless}

def select_samples(n, seed=0):
    data = json.load(open(T1))
    data = [d for d in data if os.path.exists(os.path.join(GT, d["uid"].split("/")[-1] + ".obj"))]
    rng = random.Random(seed); rng.shuffle(data)
    return data[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--model", required=True); ap.add_argument("--level", default="l1", choices=["l0", "l1", "l2", "l3"])
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--price-in", type=float, default=None); ap.add_argument("--price-out", type=float, default=None)
    a = ap.parse_args()
    samples = select_samples(a.n, a.seed)
    outdir = os.path.join(ROOT, "results", a.name.replace("|","_").replace("/","_")); os.makedirs(os.path.join(outdir, "stl"), exist_ok=True)
    gen = BACKENDS[a.backend]
    if a.backend == "serverless": _sampler(a.model)  # eager init before threads
    pin, pout = (a.price_in, a.price_out) if a.price_in is not None else PRICES.get(a.model, (0.45, 1.335) if a.model.startswith("ckpt|") else (0.0, 0.0))

    def work(s):
        uid = s["uid"].split("/")[-1]; prompt = s[f"prompt_{a.level}"]
        rec = dict(uid=uid, level=a.level, n_ops=s["n_operations"])
        try:
            text, ti, to, dt = gen(a.model, prompt, a.max_tokens, a.temperature)
        except Exception as e:
            rec.update(error=f"gen: {e}"[:300], ok=False, latency=None); return rec
        code = extract_code(text)
        rec.update(tokens_in=ti, tokens_out=to, latency=dt, code=code, raw_len=len(text))
        if not code.strip():
            rec.update(ok=False, error="empty output (truncated or no code)"); return rec
        stl = os.path.join(outdir, "stl", uid + ".stl")
        ok, err = run_cadquery(code, stl, timeout=60)
        rec.update(ok=ok, error=err)
        if ok:
            rec.update(evaluate_pair(stl, os.path.join(GT, uid + ".obj")))
        return rec

    recs = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, s) for s in samples]
        for i, f in enumerate(as_completed(futs), 1):
            recs.append(f.result())
            if i % 25 == 0: print(f"{i}/{len(samples)}", flush=True)
    with open(os.path.join(outdir, "records.jsonl"), "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    n = len(recs); oks = [r for r in recs if r.get("ok")]
    cds = [r["cd_mean"] for r in oks if r.get("cd_mean") is not None]
    ious = [r["iou"] for r in oks if r.get("iou") is not None]
    lats = [r["latency"] for r in recs if r.get("latency")]
    tin = sum(r.get("tokens_in", 0) for r in recs); tout = sum(r.get("tokens_out", 0) for r in recs)
    cost = (tin * pin + tout * pout) / 1e6
    # CD with failure penalty: failed samples count as CD=100 (worst) for a "quality@all" number
    cd_all = (sum(cds) + 100.0 * (n - len(cds))) / n if n else None
    summary = dict(name=a.name, model=a.model, level=a.level, n=n, exec_rate=len(oks) / n,
                   cd_mean=statistics.mean(cds) if cds else None, cd_median=statistics.median(cds) if cds else None,
                   cd_all_penalized=cd_all, iou_mean=statistics.mean(ious) if ious else None,
                   cd_lt_5_rate=sum(c < 5 for c in cds) / n, cd_lt_10_rate=sum(c < 10 for c in cds) / n,
                   latency_p50=statistics.median(lats) if lats else None, latency_mean=statistics.mean(lats) if lats else None,
                   tokens_in=tin, tokens_out=tout, cost_usd=cost, cost_per_1k_parts=cost / n * 1000 if n else None)
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
