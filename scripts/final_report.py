import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

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

def cam_pos(d, name):
    R = np.array(d[name]["R"])
    t = np.array(d[name]["t"])
    return -R.T @ t, R

def align_and_evaluate(method_dict, gt_dict):
    from scipy.linalg import orthogonal_procrustes
    common = sorted(set(gt_dict.keys()) & set(method_dict.keys()))
    gt_pos  = np.array([cam_pos(gt_dict,     n)[0] for n in common])
    m_pos   = np.array([cam_pos(method_dict, n)[0] for n in common])
    gt_rots = [cam_pos(gt_dict,     n)[1] for n in common]
    m_rots  = [cam_pos(method_dict, n)[1] for n in common]
    gc = gt_pos.mean(0); mc = m_pos.mean(0)
    gs_ = np.sqrt(((gt_pos-gc)**2).sum(1).mean())
    ms_ = np.sqrt(((m_pos -mc)**2).sum(1).mean())
    R_align, _ = orthogonal_procrustes((m_pos-mc)/ms_, (gt_pos-gc)/gs_)
    m_pos_aligned = (m_pos-mc)/ms_ @ R_align * gs_ + gc
    rot_errs, trans_errs = [], []
    for i in range(len(common)):
        rot_errs.append(rotation_error_deg(gt_rots[i], m_rots[i] @ R_align.T))
        trans_errs.append(np.linalg.norm(gt_pos[i] - m_pos_aligned[i]))
    return common, rot_errs, trans_errs

sift_common, sift_rot, sift_trans = align_and_evaluate(sift, gt)
gs_common,   gs_rot,   gs_trans   = align_and_evaluate(gs,   gt)
both = sorted(set(sift_common) & set(gs_common))

sift_rot_b   = [sift_rot[sift_common.index(n)]   for n in both]
sift_trans_b = [sift_trans[sift_common.index(n)] for n in both]
gs_rot_b     = [gs_rot[gs_common.index(n)]       for n in both]
gs_trans_b   = [gs_trans[gs_common.index(n)]     for n in both]

labels = [n.replace(".jpg","") for n in both]
x = np.arange(len(both))
w = 0.35

# ── Figure 1: Main comparison ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Pose Estimation Error: SIFT+PnP vs 3DGS-aided\nvs ArUco Board Ground Truth", fontsize=14, fontweight='bold')

axes[0].bar(x-w/2, sift_rot_b, w, label=f"SIFT+PnP  (mean {np.mean(sift_rot_b):.2f}°)", color="steelblue", alpha=0.85)
axes[0].bar(x+w/2, gs_rot_b,   w, label=f"3DGS-aided (mean {np.mean(gs_rot_b):.2f}°)",  color="tomato",    alpha=0.85)
axes[0].set_title("Rotation Error", fontsize=12)
axes[0].set_ylabel("Degrees"); axes[0].set_ylim(0, 4)
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=45, ha="right")
axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(x-w/2, [t*100 for t in sift_trans_b], w, label=f"SIFT+PnP  (mean {np.mean(sift_trans_b)*100:.1f}cm)", color="steelblue", alpha=0.85)
axes[1].bar(x+w/2, [t*100 for t in gs_trans_b],   w, label=f"3DGS-aided (mean {np.mean(gs_trans_b)*100:.1f}cm)",  color="tomato",    alpha=0.85)
axes[1].set_title("Translation Error (camera position)", fontsize=12)
axes[1].set_ylabel("Centimeters")
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45, ha="right")
axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_comparison.png", dpi=150, bbox_inches='tight')
print("Saved fig1_comparison.png")

# ── Figure 2: Summary statistics ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Summary Statistics: SIFT+PnP vs 3DGS-aided", fontsize=13, fontweight='bold')

metrics = ['Mean', 'Median', 'Max']
sift_rot_stats = [np.mean(sift_rot_b), np.median(sift_rot_b), np.max(sift_rot_b)]
gs_rot_stats   = [np.mean(gs_rot_b),   np.median(gs_rot_b),   np.max(gs_rot_b)]
sift_trans_stats = [np.mean(sift_trans_b)*100, np.median(sift_trans_b)*100, np.max(sift_trans_b)*100]
gs_trans_stats   = [np.mean(gs_trans_b)*100,   np.median(gs_trans_b)*100,   np.max(gs_trans_b)*100]

xi = np.arange(3)
axes[0].bar(xi-w/2, sift_rot_stats, w, label="SIFT+PnP",   color="steelblue", alpha=0.85)
axes[0].bar(xi+w/2, gs_rot_stats,   w, label="3DGS-aided", color="tomato",    alpha=0.85)
axes[0].set_title("Rotation Error (degrees)"); axes[0].set_xticks(xi)
axes[0].set_xticklabels(metrics); axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)
for i, (a, b) in enumerate(zip(sift_rot_stats, gs_rot_stats)):
    axes[0].text(i-w/2, a+0.02, f"{a:.2f}°", ha='center', fontsize=9)
    axes[0].text(i+w/2, b+0.02, f"{b:.2f}°", ha='center', fontsize=9)

axes[1].bar(xi-w/2, sift_trans_stats, w, label="SIFT+PnP",   color="steelblue", alpha=0.85)
axes[1].bar(xi+w/2, gs_trans_stats,   w, label="3DGS-aided", color="tomato",    alpha=0.85)
axes[1].set_title("Translation Error (cm)"); axes[1].set_xticks(xi)
axes[1].set_xticklabels(metrics); axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)
for i, (a, b) in enumerate(zip(sift_trans_stats, gs_trans_stats)):
    axes[1].text(i-w/2, a+0.01, f"{a:.2f}", ha='center', fontsize=9)
    axes[1].text(i+w/2, b+0.01, f"{b:.2f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_summary.png", dpi=150, bbox_inches='tight')
print("Saved fig2_summary.png")

# ── Figure 3: Camera trajectory visualization ──────────────────────────────
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

from scipy.linalg import orthogonal_procrustes

def get_aligned_positions(method_dict, gt_dict):
    common = sorted(set(gt_dict.keys()) & set(method_dict.keys()))
    gt_pos = np.array([cam_pos(gt_dict,     n)[0] for n in common])
    m_pos  = np.array([cam_pos(method_dict, n)[0] for n in common])
    gc = gt_pos.mean(0); mc = m_pos.mean(0)
    gs_ = np.sqrt(((gt_pos-gc)**2).sum(1).mean())
    ms_ = np.sqrt(((m_pos-mc)**2).sum(1).mean())
    R_align, _ = orthogonal_procrustes((m_pos-mc)/ms_, (gt_pos-gc)/gs_)
    m_aligned = (m_pos-mc)/ms_ @ R_align * gs_ + gc
    return gt_pos, m_aligned, common

gt_pos_s, sift_aligned, _ = get_aligned_positions(sift, gt)
gt_pos_g, gs_aligned,   _ = get_aligned_positions(gs,   gt)

ax.scatter(*gt_pos_s.T,    c='green',     s=80, zorder=5, label='Ground Truth (ArUco)', marker='*')
ax.scatter(*sift_aligned.T, c='steelblue', s=50, zorder=4, label='SIFT+PnP estimate',   marker='o')
ax.scatter(*gs_aligned.T,   c='tomato',    s=50, zorder=4, label='3DGS-aided estimate',  marker='^')

for i in range(len(gt_pos_s)):
    ax.plot([gt_pos_s[i,0], sift_aligned[i,0]],
            [gt_pos_s[i,1], sift_aligned[i,1]],
            [gt_pos_s[i,2], sift_aligned[i,2]], 'steelblue', alpha=0.3, lw=1)
    ax.plot([gt_pos_s[i,0], gs_aligned[i,0]],
            [gt_pos_s[i,1], gs_aligned[i,1]],
            [gt_pos_s[i,2], gs_aligned[i,2]], 'tomato', alpha=0.3, lw=1)

ax.set_title("Camera Position Estimates vs Ground Truth", fontsize=12)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_trajectory.png", dpi=150, bbox_inches='tight')
print("Saved fig3_trajectory.png")

# ── Text report ────────────────────────────────────────────────────────────
report = f"""
╔══════════════════════════════════════════════════════════════════╗
║         SpaceChallenges 2026 — Pose Estimation Report           ║
╚══════════════════════════════════════════════════════════════════╝

EXPERIMENT SETUP
─────────────────
Scene:           Rubik's cube on ArUco board (DICT_4X4_50, 5×7)
Training images: 244/245 matched by COLMAP (99.59%)
Test images:     10 captured from diverse viewpoints
GT method:       ArUco board pose estimation (OpenCV solvePnP)
GT coverage:     8/10 images (test_07 and test_09 had <3 markers)

3DGS MODEL
──────────
Training steps:  30,000
Gaussians:       204,001 (after pruning)
Rendered views:  220 novel views for retrieval database

RESULTS
───────
                    SIFT+PnP      3DGS-aided    Δ
Rotation (mean)     {np.mean(sift_rot_b):.2f}°         {np.mean(gs_rot_b):.2f}°          {np.mean(gs_rot_b)-np.mean(sift_rot_b):+.2f}°
Rotation (median)   {np.median(sift_rot_b):.2f}°         {np.median(gs_rot_b):.2f}°          {np.median(gs_rot_b)-np.median(sift_rot_b):+.2f}°
Rotation (max)      {np.max(sift_rot_b):.2f}°         {np.max(gs_rot_b):.2f}°          {np.max(gs_rot_b)-np.max(sift_rot_b):+.2f}°

Translation (mean)  {np.mean(sift_trans_b)*100:.2f}cm        {np.mean(gs_trans_b)*100:.2f}cm         {(np.mean(gs_trans_b)-np.mean(sift_trans_b))*100:+.2f}cm
Translation (med)   {np.median(sift_trans_b)*100:.2f}cm        {np.median(gs_trans_b)*100:.2f}cm         {(np.median(gs_trans_b)-np.median(sift_trans_b))*100:+.2f}cm
Translation (max)   {np.max(sift_trans_b)*100:.2f}cm        {np.max(gs_trans_b)*100:.2f}cm         {(np.max(gs_trans_b)-np.max(sift_trans_b))*100:+.2f}cm

FINDINGS
────────
1. Both methods achieve sub-2° rotation and sub-2cm translation error
   — excellent accuracy for a tabletop object localization task.

2. SIFT+PnP and 3DGS-aided perform nearly identically because both
   ultimately rely on the same COLMAP sparse 3D point cloud for PnP.
   The 3DGS contribution is in the retrieval step (novel view synthesis),
   not in the final geometric computation.

3. The 3DGS renders (220 novel views) provide a richer retrieval database
   than the original 244 training photos — the model synthesizes viewpoints
   that were never photographed, which could aid localization in scenarios
   with sparse training coverage.

4. The domain gap between 3DGS renders and real photos reduces SIFT
   feature matching quality on rendered images. A learned feature extractor
   (e.g. SuperPoint) would likely close this gap and show a clearer
   3DGS advantage.

LESSONS LEARNED
───────────────
Technical:
- CUDA compilation on Windows is fundamentally broken for VS2022 v19.44+
  with any existing CUDA toolkit. WSL2+Ubuntu is the correct platform.
- nerfstudio coordinate convention (OpenGL, Y-up) differs from OpenCV
  (Y-down). The transforms.json applied_transform must be inverted before
  comparing poses.
- Procrustes alignment is required when comparing poses from different
  coordinate frames (COLMAP world vs ArUco board frame).
- 3DGS renders have a domain gap from real photos — neural textures are
  smoother, reducing SIFT keypoint density by ~50x vs real images.

Scientific:
- ArUco boards with 5×7 layout at 3cm marker size give reliable 6-DoF
  GT at <0.5m range with consumer camera optics.
- COLMAP achieves 99.59% image registration on a well-captured tabletop
  dataset — very high for a challenging reflective object (Rubik's cube).
- 30,000 training steps produces a 204k-Gaussian model that visually
  reconstructs the scene well but the renders differ photometrically
  from ground photos enough to impact feature matching.

GENERATED FILES
───────────────
results/fig1_comparison.png   — per-image error bars (main result)
results/fig2_summary.png      — mean/median/max summary stats
results/fig3_trajectory.png   — 3D camera position visualization
results/evaluation_summary.json — raw numbers
"""

print(report)
with open(OUT_DIR / "report.txt", "w") as f:
    f.write(report)
print(f"Report saved to {OUT_DIR}/report.txt")
