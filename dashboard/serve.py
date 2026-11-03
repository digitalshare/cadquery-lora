"""Live demo server: serves dashboard/index.html with window.LIVE=true and runs prompts against
the untuned base, the tuned checkpoint, and Claude Opus 5.5.

.venv/bin/python dashboard/serve.py   ->  http://localhost:8787
"""
import base64, json, os, sys, tempfile, time, threading
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval")); sys.path.insert(0, os.path.join(ROOT, "demo"))
os.environ.setdefault("TINKER_TELEMETRY", "0")
from flask import Flask, request, jsonify, Response
import run as R
from cadexec import run_cadquery
from compare import render_png

state = json.load(open(os.path.join(ROOT, "results/serverless/state-v1.json")))
CKPT = [c["path"] for c in state["checkpoints"] if c["name"] == "v1-s100"][0]
TOK = state["tokenizer"]
MODELS = {
    "base": ("serverless", f"base|{state['base_model']}|{TOK}"),
    "tuned": ("serverless", f"ckpt|{CKPT}|{TOK}"),
    "opus": ("anthropic", "claude-opus-5-5"),
}
PRICES = {"base": (0.45, 1.335), "tuned": (0.45, 1.335), "opus": (4.0, 20.0)}
app = Flask(__name__)
html = open(os.path.join(ROOT, "dashboard/index.html")).read().replace("window.LIVE=false", "window.LIVE=true")

@app.get("/")
def index(): return Response(html, mimetype="text/html")

@app.post("/run")
def run_one():
    body = request.get_json(force=True); key = body["model"]; prompt = body["prompt"]
    backend, model = MODELS[key]
    try:
        text, ti, to, dt = R.BACKENDS[backend](model, prompt, 2048)
    except Exception as e:
        return jsonify(dict(model=key, ok=False, error=f"generation failed: {e}"[:300]))
    code = R.extract_code(text)
    stl = tempfile.NamedTemporaryFile(suffix=".stl", delete=False).name
    ok, err = run_cadquery(code, stl, timeout=60)
    png = None
    if ok:
        p = stl.replace(".stl", ".png")
        try: render_png(stl, p); png = "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
        except Exception as e: err = f"render failed: {e}"[:200]
    pin, pout = PRICES[key]
    return jsonify(dict(model=key, ok=ok, error=err, code=code, latency=dt, tokens_in=ti, tokens_out=to,
                        cost_usd=(ti * pin + to * pout) / 1e6, png=png))

def warm():
    for k in ("base", "tuned"):
        try: R._sampler(MODELS[k][1]); print("warm:", k, flush=True)
        except Exception as e: print("warm failed", k, e, flush=True)

if __name__ == "__main__":
    threading.Thread(target=warm, daemon=True).start()
    app.run(host="127.0.0.1", port=8787, debug=False, threaded=True)
