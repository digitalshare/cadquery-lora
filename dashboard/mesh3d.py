"""Serialize a mesh (STL/OBJ) into a compact payload for the dashboard's three.js viewer.

Payload: {"pos": <base64 float32 xyz, 3 vertices per triangle, non-indexed>, "n": <triangle count>}
Meshes are centered and scaled to fit a unit box so every pane shows the same scale.
"""
import base64, numpy as np, trimesh

MAX_FACES = 40000

def mesh_payload(path):
    m = trimesh.load(path, force="mesh")
    if m is None or m.is_empty or len(m.faces) == 0:
        return None
    b = m.bounds; scale = float((b[1] - b[0]).max()) or 1.0
    v = (m.vertices - (b[0] + b[1]) / 2.0) / scale
    f = m.faces
    if len(f) > MAX_FACES:  # keep payloads small: uniform random face subsample is wrong, so stride instead
        f = f[:: int(np.ceil(len(f) / MAX_FACES))]
    tri = v[f].reshape(-1, 3).astype(np.float32)  # (n*3, 3)
    return {"pos": base64.b64encode(tri.tobytes()).decode(), "n": int(len(f))}
