"""Chamfer distance / IoU identical to TextCAD-Bench protocol (eval/metrics.py upstream)."""
import numpy as np, trimesh
from scipy.spatial import cKDTree

def load_and_normalize(path):
    mesh = trimesh.load(path, force='mesh')
    if mesh is None or mesh.is_empty or len(mesh.vertices) == 0:
        return None
    b = mesh.bounds; scale = (b[1] - b[0]).max()
    if scale < 1e-8: return None
    mesh.apply_translation(-(b[0] + b[1]) / 2); mesh.apply_scale(1.0 / scale)
    return mesh

def chamfer_distance(m1, m2, n_points=8192, seed=0):
    p1, _ = trimesh.sample.sample_surface(m1, n_points, seed=seed)
    p2, _ = trimesh.sample.sample_surface(m2, n_points, seed=seed)
    d1 = cKDTree(p2).query(p1)[0]; d2 = cKDTree(p1).query(p2)[0]
    return float(np.mean(d1) + np.mean(d2)) * 100, float(np.median(d1) + np.median(d2)) * 100

def voxel_iou(m1, m2, resolution=64):
    try:
        v1 = m1.voxelized(pitch=1.0 / resolution).fill().matrix
        v2 = m2.voxelized(pitch=1.0 / resolution).fill().matrix
        s = tuple(max(a, b) for a, b in zip(v1.shape, v2.shape))
        a = np.zeros(s, bool); b = np.zeros(s, bool)
        a[:v1.shape[0], :v1.shape[1], :v1.shape[2]] = v1; b[:v2.shape[0], :v2.shape[1], :v2.shape[2]] = v2
        inter = (a & b).sum(); union = (a | b).sum()
        return float(inter) / float(union) * 100 if union > 0 else 0.0
    except Exception:
        return None

def evaluate_pair(pred_stl, gt_obj):
    out = dict(cd_mean=None, cd_median=None, iou=None, watertight=False)
    pred = load_and_normalize(pred_stl); gt = load_and_normalize(gt_obj)
    if pred is None or gt is None: return out
    out['watertight'] = bool(pred.is_watertight)
    try: out['cd_mean'], out['cd_median'] = chamfer_distance(pred, gt)
    except Exception: pass
    if pred.is_watertight and gt.is_watertight:
        out['iou'] = voxel_iou(pred, gt)
    return out
