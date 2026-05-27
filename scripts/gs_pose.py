import json
import numpy as np
import cv2
from pathlib import Path

# 3DGS localization via image retrieval:
# For each test image, find the most visually similar rendered view
# from the 3DGS model and use its known pose as the estimate.
# This demonstrates the 3DGS model's novel view synthesis for localization.

PROJECT = Path("/mnt/c/Users/i3pur1la/Desktop/SpaceChallenges")
TEST_IMAGES = PROJECT / "data/test/images"
TRANSFORMS  = PROJECT / "reconstruction/colmap_out/transforms.json"
RENDER_DIR  = PROJECT / "reconstruction/gs_renders/train/rgb"
OUTPUT      = PROJECT / "data/test/gs_pose_results.json"

with open(TRANSFORMS) as f:
    transforms = json.load(f)

fx = transforms["fl_x"]; fy = transforms["fl_y"]
cx = transforms["cx"];   cy = transforms["cy"]
k1 = transforms.get("k1",0); k2 = transforms.get("k2",0)
p1 = transforms.get("p1",0); p2 = transforms.get("p2",0)

camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
dist_coeffs   = np.array([k1,k2,p1,p2], dtype=np.float64)

frames_ordered = [Path(f["file_path"]).name for f in transforms["frames"]]
frame_lookup = {Path(f["file_path"]).name: np.array(f["transform_matrix"], dtype=np.float64)
                for f in transforms["frames"]}

def c2w_to_w2c(c2w):
    c2w_cv = c2w.copy()
    c2w_cv[:3,1:3] *= -1
    w2c = np.linalg.inv(c2w_cv)
    return w2c[:3,:3], w2c[:3,3]

def load_ply(path):
    pts = []
    with open(path,'r') as f:
        header = True
        for line in f:
            if header:
                if line.strip()=='end_header': header=False
                continue
            v = line.strip().split()
            if len(v)>=3: pts.append([float(x) for x in v[:3]])
    return np.array(pts, dtype=np.float32)

points_3d = load_ply(PROJECT / "reconstruction/colmap_out/sparse_pc.ply")

renders = sorted(RENDER_DIR.glob("*.jpg"))
print(f"Found {len(renders)} 3DGS rendered views")

def global_descriptor(img, size=(64,64)):
    """Simple but effective global descriptor: resized + color histogram."""
    img_small = cv2.resize(img, size)
    # HSV histogram
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv],[0],None,[32],[0,180]).flatten()
    hist_s = cv2.calcHist([hsv],[1],None,[32],[0,256]).flatten()
    hist_v = cv2.calcHist([hsv],[2],None,[32],[0,256]).flatten()
    desc = np.concatenate([hist_h, hist_s, hist_v])
    desc /= (desc.sum() + 1e-7)
    return desc

# Build render descriptor database
print("Computing descriptors for all rendered views...")
render_descs  = []
render_frames = []

for render_path in renders:
    num = int(''.join(filter(str.isdigit, render_path.stem))) - 1
    if num < 0 or num >= len(frames_ordered):
        continue
    frame_name = frames_ordered[num]
    frame_data = frame_lookup.get(frame_name)
    if frame_data is None:
        continue

    img = cv2.imread(str(render_path))
    if img is None: continue

    desc = global_descriptor(img)
    render_descs.append(desc)
    render_frames.append((frame_name, frame_data, render_path))

render_descs = np.array(render_descs, dtype=np.float32)
print(f"Built descriptor database for {len(render_descs)} renders")

# Now also build a SIFT database from training images for PnP refinement
print("Building SIFT database from training images for pose refinement...")
sift = cv2.SIFT_create(nfeatures=5000)
train_img_dir = PROJECT / "reconstruction/colmap_out/images"
db_desc, db_pts3d = [], []

for idx, img_path in enumerate(sorted(train_img_dir.glob("*.jpg"))[:100]):
    frame_data = frame_lookup.get(img_path.name)
    if frame_data is None: continue

    img = cv2.imread(str(img_path))
    if img is None: continue
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    kps, descs = sift.detectAndCompute(img_gray, None)
    if descs is None: continue

    R, t = c2w_to_w2c(frame_data)
    pts_cam = (R @ points_3d.T).T + t
    valid = pts_cam[:,2] > 0
    pts_cam_v = pts_cam[valid]; pts_3d_v = points_3d[valid]
    if len(pts_3d_v)==0: continue

    pts_proj = pts_cam_v[:,:2] / pts_cam_v[:,2:3]
    pts_proj[:,0] = pts_proj[:,0]*fx + cx
    pts_proj[:,1] = pts_proj[:,1]*fy + cy
    in_f = ((pts_proj[:,0]>=0)&(pts_proj[:,0]<w)&
            (pts_proj[:,1]>=0)&(pts_proj[:,1]<h))
    pts_proj_f = pts_proj[in_f]; pts_3d_f = pts_3d_v[in_f]
    if len(pts_proj_f)==0: continue

    for kp, desc in zip(kps, descs):
        dists = np.linalg.norm(pts_proj_f - np.array(kp.pt), axis=1)
        nearest = np.argmin(dists)
        if dists[nearest] < 5.0:
            db_desc.append(desc)
            db_pts3d.append(pts_3d_f[nearest])

db_desc  = np.array(db_desc,  dtype=np.float32)
db_pts3d = np.array(db_pts3d, dtype=np.float32)
print(f"SIFT database: {len(db_desc)} correspondences")

flann = cv2.FlannBasedMatcher(dict(algorithm=1,trees=5), dict(checks=100))
results = {}

for img_path in sorted(TEST_IMAGES.glob("*.jpg")):
    img_bgr  = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step 1: Find nearest rendered view using global descriptor
    test_desc = global_descriptor(img_bgr)
    dists = np.linalg.norm(render_descs - test_desc, axis=1)
    top5 = np.argsort(dists)[:5]

    best_name, best_frame, best_render = render_frames[top5[0]]
    similarity_score = float(dists[top5[0]])

    # Step 2: Get pose from nearest render's known camera pose
    R_init, t_init = c2w_to_w2c(best_frame)

    # Step 3: Refine with SIFT+PnP
    kps, descs = sift.detectAndCompute(img_gray, None)
    R_final, t_final, n_inliers = R_init, t_init, 0

    if descs is not None and len(descs) >= 10:
        matches = flann.knnMatch(descs.astype(np.float32), db_desc, k=2)
        good = [m for m,n in matches if m.distance < 0.7*n.distance]

        if len(good) >= 6:
            pts_2d   = np.array([kps[m.queryIdx].pt   for m in good], dtype=np.float32)
            pts_3d_m = np.array([db_pts3d[m.trainIdx] for m in good], dtype=np.float32)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d_m, pts_2d, camera_matrix, dist_coeffs,
                reprojectionError=8.0, confidence=0.99, iterationsCount=1000
            )
            if success and inliers is not None and len(inliers) >= 4:
                R_final, _ = cv2.Rodrigues(rvec)
                t_final = tvec.flatten()
                n_inliers = len(inliers)

    results[img_path.name] = {
        "R": R_final.tolist(),
        "t": t_final.tolist() if hasattr(t_final, 'tolist') else list(t_final),
        "nearest_render": best_name,
        "retrieval_score": similarity_score,
        "num_inliers": n_inliers
    }
    print(f"[OK] {img_path.name}: nearest={best_name}, inliers={n_inliers}, score={similarity_score:.4f}")

with open(OUTPUT,"w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved 3DGS-aided results for {len(results)}/10 images")
