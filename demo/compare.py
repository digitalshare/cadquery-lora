"""Side-by-side demo: one description -> CadQuery code + rendered PNG from several models.

python demo/compare.py "A flat circular flange with a central hole and four bolt holes" \
    --model fireworks:accounts/fireworks/models/gpt-oss-120b --model anthropic:claude-opus-5-5
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
import run as R
from cadexec import run_cadquery

def render_png(stl, png):
    import trimesh, numpy as np
    m = trimesh.load(stl, force="mesh")
    try:
        scene = m.scene(); scene.set_camera(angles=(0.9, 0.0, 0.6), distance=m.scale * 2.2)
        data = scene.save_image(resolution=(480, 400), visible=False)
        open(png, "wb").write(data); return True
    except Exception:
        # headless fallback: matplotlib tri-surface
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(4.8, 4)); ax = fig.add_subplot(111, projection="3d")
        v, f = m.vertices, m.faces
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color=(0.3, 0.55, 0.9, 1.0), edgecolor="k", linewidth=0.1)
        ax.set_box_aspect((np.ptp(v[:, 0]) or 1, np.ptp(v[:, 1]) or 1, np.ptp(v[:, 2]) or 1)); ax.set_axis_off()
        fig.tight_layout(); fig.savefig(png, dpi=100); plt.close(fig); return True

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prompt"); ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out", default="demo/out"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rows = []
    for spec in a.model:
        backend, model = spec.split(":", 1); tag = model.split("/")[-1].replace("#", "_")
        t = time.time(); text, ti, to, dt = R.BACKENDS[backend](model, a.prompt, 4000)
        code = R.extract_code(text); stl = os.path.join(a.out, tag + ".stl"); png = os.path.join(a.out, tag + ".png")
        ok, err = run_cadquery(code, stl, timeout=60)
        if ok: render_png(stl, png)
        open(os.path.join(a.out, tag + ".py"), "w").write(code)
        rows.append(dict(model=model, ok=ok, err=err, latency=round(dt, 2), tokens_out=to, png=png if ok else None))
        print(f"{tag:40s} ok={ok} {dt:.1f}s out={to} {err[:80]}")
    json.dump(rows, open(os.path.join(a.out, "summary.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
