"""LoRA SFT on Fireworks Serverless Training (own loop, SDK only).

python train/sft_serverless.py --max-examples 12000 --batch 16 --accum 2 --lr 1e-4 --rank 16 --tag v1
"""
import argparse, json, math, os, random, sys, time
import numpy as np
from transformers import AutoTokenizer
from fireworks.training import sdk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "https://api.fireworks.ai/training/v1/serverless"

def render(tok, messages, max_len):
    """Tokens for the whole conversation + float weights (1/n_resp on assistant tokens, 0 elsewhere)."""
    full = tok.apply_chat_template(messages, tokenize=False)
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    assert full.startswith(prompt), "chat template prefix mismatch"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if f_ids[:len(p_ids)] != p_ids:  # tokenization boundary drift; re-tokenize response separately
        r_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]; f_ids = p_ids + r_ids
    if len(f_ids) > max_len or len(f_ids) - len(p_ids) < 1:
        return None
    w = np.zeros(len(f_ids), dtype=np.float32); w[len(p_ids):] = 1.0 / (len(f_ids) - len(p_ids))
    return f_ids, w

def to_datum(tokens, weights):
    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": sdk.TensorData(data=tokens[1:], dtype="int64", shape=[len(tokens) - 1]),
            "weights": sdk.TensorData(data=weights[1:].tolist(), dtype="float32", shape=[len(tokens) - 1]),
        },
    )

def batch_loss(batch, out):
    """Weighted NLL from per-token logprobs (weights already sum to 1 per example)."""
    tot = 0.0; n = 0
    for d, o in zip(batch, out.loss_fn_outputs):
        lp = np.asarray(o["logprobs"].to_numpy() if hasattr(o["logprobs"], "to_numpy") else o["logprobs"].data, dtype=np.float64)
        w = np.asarray(d.loss_fn_inputs["weights"].data, dtype=np.float64)
        if len(lp) != len(w): continue
        tot += float(-(lp * w).sum()); n += 1
    return tot / max(n, 1)

def retry(fn, tries=4, what=""):
    for i in range(tries):
        try: return fn()
        except Exception as e:
            if i == tries - 1: raise
            print(f"  retry {what} ({i+1}): {str(e)[:160]}", flush=True); time.sleep(5 * (i + 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="accounts/fireworks/models/qwen3p6-35b-a3b")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--data", default=os.path.join(ROOT, "data/train.jsonl"))
    ap.add_argument("--max-examples", type=int, default=12000); ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=16, help="datums per forward_backward call")
    ap.add_argument("--accum", type=int, default=2, help="forward_backward calls per optim_step")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--tag", default="v1")
    ap.add_argument("--save-every", type=int, default=150, help="optimizer steps between sampler checkpoints")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, "results/serverless"); os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, f"state-{a.tag}.json")

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [json.loads(l) for l in open(a.data)]
    random.Random(a.seed).shuffle(rows); rows = rows[:a.max_examples]
    rendered = []; dropped = 0
    for r in rows:
        x = render(tok, r["messages"], a.max_len)
        if x is None: dropped += 1
        else: rendered.append(x)
    n_tok = sum(len(t) for t, _ in rendered)
    print(f"examples {len(rendered)} (dropped {dropped}), train tokens {n_tok/1e6:.2f}M, est cost ${n_tok/1e6*2.14:.2f}", flush=True)
    steps_total = math.ceil(len(rendered) / (a.batch * a.accum))
    print(f"optimizer steps: {steps_total} (batch {a.batch} x accum {a.accum})", flush=True)
    if a.dry_run:
        d = to_datum(*rendered[0]); print("datum ok:", d.model_input.length, len(d.loss_fn_inputs["target_tokens"].data)); return

    service = sdk.FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=BASE_URL)
    tc = service.create_lora_training_client(base_model=a.base_model, rank=a.rank, seed=a.seed)
    state = dict(tag=a.tag, base_model=a.base_model, tokenizer=a.tokenizer, session=service.training_session_name,
                 session_id=service.training_session_id, run_id=tc.run_id, checkpoints=[], steps=[], args=vars(a), started=time.time())
    json.dump(state, open(state_path, "w"), indent=2)
    print(f"session={service.training_session_name} run={tc.run_id}", flush=True)

    def lr_at(step):
        if step < a.warmup: return a.lr * (step + 1) / a.warmup
        p = (step - a.warmup) / max(1, steps_total - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    idx = 0; step = 0; t0 = time.time(); losses = []
    while idx < len(rendered):
        step_loss = []
        for _ in range(a.accum):
            chunk = rendered[idx: idx + a.batch]; idx += a.batch
            if not chunk: break
            batch = [to_datum(t, w) for t, w in chunk]
            out = retry(lambda: tc.forward_backward(batch, "cross_entropy").result(timeout=600), what="fwd_bwd")
            step_loss.append(batch_loss(batch, out))
        retry(lambda: tc.optim_step(sdk.AdamParams(learning_rate=lr_at(step), beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)).result(timeout=600), what="optim")
        step += 1; l = float(np.mean(step_loss)); losses.append(l)
        el = time.time() - t0; eta = el / step * (steps_total - step)
        print(f"step {step}/{steps_total} loss {l:.4f} lr {lr_at(step-1):.2e} elapsed {el/60:.1f}m eta {eta/60:.1f}m", flush=True)
        state["steps"].append(dict(step=step, loss=l, t=el))
        if step % a.save_every == 0 or idx >= len(rendered):
            name = f"{a.tag}-s{step}"[:17]
            path = retry(lambda: tc.save_weights_for_sampler(name).result(timeout=900).path, what="save")
            state["checkpoints"].append(dict(step=step, name=name, path=path, loss=l)); print(f"  saved sampler checkpoint {name} -> {path}", flush=True)
        json.dump(state, open(state_path, "w"), indent=2)
    state["finished"] = time.time(); json.dump(state, open(state_path, "w"), indent=2)
    print("done. session kept alive for eval/promotion:", service.training_session_name, flush=True)

if __name__ == "__main__":
    main()
