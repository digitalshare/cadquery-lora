"""Execute CadQuery code safely in a subprocess and export an STL.

Usage as a library:
    from cadexec import run_cadquery
    ok, err = run_cadquery(code, "/tmp/out.stl", timeout=30)
"""
import os, subprocess, sys, tempfile, textwrap

PY = sys.executable  # venv python with cadquery installed

RUNNER = textwrap.dedent(r'''
import sys, json
code_path, out_path = sys.argv[1], sys.argv[2]
src = open(code_path).read()
import cadquery as cq
ns = {"__name__": "__main__"}
captured = []
def show_object(obj, *a, **k):
    captured.append(obj)
ns["show_object"] = show_object
exec(compile(src, "<gen>", "exec"), ns)
obj = ns.get("result")
if obj is None and captured:
    obj = captured[-1]
if obj is None:
    # fall back: last Workplane / Shape in namespace
    for v in reversed(list(ns.values())):
        if isinstance(v, (cq.Workplane, cq.Shape, cq.Assembly)):
            obj = v; break
if obj is None:
    raise RuntimeError("no result object found")
if isinstance(obj, cq.Assembly):
    obj = obj.toCompound()
if isinstance(obj, cq.Workplane):
    vals = obj.vals()
    if not vals:
        raise RuntimeError("empty workplane")
    shape = cq.Compound.makeCompound([v for v in vals if isinstance(v, cq.Shape)]) if len(vals) > 1 else vals[0]
else:
    shape = obj
if not isinstance(shape, cq.Shape):
    raise RuntimeError(f"result is {type(shape)}")
if shape.Volume() <= 1e-9:
    raise RuntimeError("zero-volume result")
cq.exporters.export(shape, out_path, tolerance=0.01, angularTolerance=0.1)
''')

_runner_path = None
def _runner():
    global _runner_path
    if _runner_path is None:
        d = tempfile.mkdtemp(prefix="cadrunner_")
        _runner_path = os.path.join(d, "runner.py")
        with open(_runner_path, "w") as f:
            f.write(RUNNER)
    return _runner_path

def run_cadquery(code: str, out_stl: str, timeout: float = 30.0):
    """Returns (ok: bool, error: str)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); code_path = f.name
    try:
        p = subprocess.run([PY, "-I", _runner(), code_path, out_stl], capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            err = (p.stderr.strip().splitlines() or ["unknown error"])[-1]
            return False, err[:300]
        if not os.path.exists(out_stl) or os.path.getsize(out_stl) == 0:
            return False, "no stl written"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        try: os.unlink(code_path)
        except OSError: pass

if __name__ == "__main__":
    demo = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 20, 5).faces('>Z').hole(4)\nshow_object(result)\n"
    ok, err = run_cadquery(demo, "/tmp/_cadexec_demo.stl")
    print("ok" if ok else f"fail: {err}")
