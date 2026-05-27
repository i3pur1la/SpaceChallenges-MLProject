import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import orthogonal_procrustes

PROJECT = Path("/mnt/c/Users/i3pur1la/Desktop/SpaceChallenges")
GT_FILE   = PROJECT / "data/test/ground_truth.json"
SIFT_FILE = PROJECT / "data/test/sift_pnp_results.json"
GS_FILE   = PROJECT / "data/test/gs_pose_results.json"
OUT_DIR   = PROJECT / "results"
OUT_DIR.mkdir(exist_ok=True)

with open(GT_FILE)   as f: gt   = json.load(f)
with open(SIFT_FILE) as f: sift = json.load(f)
with open(GS_FILE)   as f: gs   = json.load(f)

def rotation_error_deg(R1, R2):
    R_rel = np.array(R1) @ np.array(R2).T
    trace = np.clip((np.trace(R_rel)-1)/2, -1, 1)
    return np.degrees(np.arccos(trace))

def align_and_evaluate(method_dict, gt_dict):
    common = sorted(set(gt_dict.keys()) & set(method_dict.keys()))
    
    def cam_pos(d, name):
        R = np.array(d[name]["R"])
        t = np.array(d[name]["t"])
        return -R.T @ t, R

    gt_pos   = np.array([cam_pos(gt_dict,     n)[0] for n in common])
    m_pos    = np.array([cam_pos(method_dict,  n)[0] for n in common])
    gt_rots  = [cam_pos(gt_dict,     n)[1] for n in common]
    m_rots   = [cam_pos(method_dict,  n)[1] for n in common]

    gc = gt_pos.mean(0); mc = m_pos.mean(0)
    gs_ = np.sqrt(((gt_pos-gc)**2).sum(1).mean())
    ms_ = np.sqrt(((m_pos -mc)**2).sum(1).mean())

    R_align, _ = orthogonal_procrustes((m_pos-mc)/ms_, (gt_pos-gc)/gs_)
    scale = gs_/ms_
    m_pos_aligned = (m_pos-mc)/ms_ @ R_align * gs_ + gc

    rot_errs   = []
    trans_errs = []
    for i, name in enumerate(common):
        R_aligned = m_rots[i] @ R_align.T
        rot_errs.append(rotation_error_deg(gt_rots[i], R_aligned))
        trans_errs.append(np.linalg.norm(gt_pos[i] - m_pos_aligned[i]))

    return common, rot_errs, trans_errs

gt_common = sorted(gt.keys())
sift_common, sift_rot, sift_trans = align_and_evaluate(sift, gt)
gs_common,   gs_rot,   gs_trans   = align_and_evaluate(gs,   gt)

# Only compare images that both methods evaluated
both = sorted(set(sift_common) & set(gs_common))
sift_rot_b   = [sift_rot[sift_common.index(n)]   for n in both]
sift_trans_b = [sift_trans[sift_common.index(n)] for n in both]
gs_rot_b     = [gs_rot[gs_common.index(n)]       for n in both]
gs_trans_b   = [gs_trans[gs_common.index(n)]     for n in both]

print(f"Comparing {len(both)} images with GT, SIFT+PnP and 3DGS results\n")
print(f"{'Image':<15} {'SIFT rot':>10} {'GS rot':>10} {'SIFT trans':>12} {'GS trans':>12}")
print("-"*60)
for i, name in enumerate(both):
    print(f"{name:<15} {sift_rot_b[i]:>9.2f}° {gs_rot_b[i]:>9.2f}° "
          f"{sift_trans_b[i]:>11.4f}m {gs_trans_b[i]:>11.4f}m")

print("-"*60)
print(f"{'Mean':<15} {np.mean(sift_rot_b):>9.2f}° {np.mean(gs_rot_b):>9.2f}° "
      f"{np.mean(sift_trans_b):>11.4f}m {np.mean(gs_trans_b):>11.4f}m")
print(f"{'Median':<15} {np.median(sift_rot_b):>9.2f}° {np.median(gs_rot_b):>9.2f}° "
      f"{np.median(sift_trans_b):>11.4f}m {np.median(gs_trans_b):>11.4f}m")
print(f"{'Max':<15} {np.max(sift_rot_b):>9.2f}° {np.max(gs_rot_b):>9.2f}° "
      f"{np.max(sift_trans_b):>11.4f}m {np.max(gs_trans_b):>11.4f}m")

# Save summary
summary = {
    "images_evaluated": both,
    "sift_pnp": {
        "rotation_error_deg":   {"mean": float(np.mean(sift_rot_b)),   "median": float(np.median(sift_rot_b)),   "max": float(np.max(sift_rot_b)),   "per_image": dict(zip(both, [float(e) for e in sift_rot_b]))},
        "translation_error_m":  {"mean": float(np.mean(sift_trans_b)), "median": float(np.median(sift_trans_b)), "max": float(np.max(sift_trans_b)), "per_image": dict(zip(both, [float(e) for e in sift_trans_b]))}
    },
    "3dgs": {
        "rotation_error_deg":   {"mean": float(np.mean(gs_rot_b)),   "median": float(np.median(gs_rot_b)),   "max": float(np.max(gs_rot_b)),   "per_image": dict(zip(both, [float(e) for e in gs_rot_b]))},
        "translation_error_m":  {"mean": float(np.mean(gs_trans_b)), "median": float(np.median(gs_trans_b)), "max": float(np.max(gs_trans_b)), "per_image": dict(zip(both, [float(e) for e in gs_trans_b]))}
    }
}

with open(OUT_DIR / "evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Plot
x = np.arange(len(both))
labels = [n.replace(".jpg","") for n in both]
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Pose Estimation Error: SIFT+PnP vs 3DGS\n(vs ArUco Ground Truth)", fontsize=14)

axes[0].bar(x-w/2, sift_rot_b, w, label=f"SIFT+PnP (mean {np.mean(sift_rot_b):.2f}°)", color="steelblue")
axes[0].bar(x+w/2, gs_rot_b,   w, label=f"3DGS     (mean {np.mean(gs_rot_b):.2f}°)",   color="tomato")
axes[0].set_title("Rotation Error")
axes[0].set_ylabel("Degrees")
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=45, ha="right")
axes[0].legend()

axes[1].bar(x-w/2, sift_trans_b, w, label=f"SIFT+PnP (mean {np.mean(sift_trans_b)*100:.1f}cm)", color="steelblue")
axes[1].bar(x+w/2, gs_trans_b,   w, label=f"3DGS     (mean {np.mean(gs_trans_b)*100:.1f}cm)",   color="tomato")
axes[1].set_title("Translation Error (camera position)")
axes[1].set_ylabel("Meters")
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45, ha="right")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "comparison.png", dpi=150)
print(f"\nPlot saved to {OUT_DIR}/comparison.png")
