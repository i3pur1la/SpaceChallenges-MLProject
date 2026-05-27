import cv2
import numpy as np
import json
from pathlib import Path

PROJECT = Path("/mnt/c/Users/i3pur1la/Desktop/SpaceChallenges")
TEST_IMAGES = PROJECT / "data/test/images"
TRANSFORMS = PROJECT / "reconstruction/colmap_out/transforms.json"
OUTPUT = PROJECT / "data/test/sift_pnp_results.json"

# Load intrinsics
with open(TRANSFORMS) as f:
    transforms = json.load(f)

fx = transforms["fl_x"]
fy = transforms["fl_y"]
cx = transforms["cx"]
cy = transforms["cy"]
k1 = transforms.get("k1", 0)
k2 = transforms.get("k2", 0)
p1 = transforms.get("p1", 0)
p2 = transforms.get("p2", 0)

camera_matrix = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float64)
dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64)

# Load sparse point cloud
def load_ply(path):
    points, colors = [], []
    with open(path, 'r') as f:
        header = True
        for line in f:
            if header:
                if line.strip() == 'end_header':
                    header = False
                continue
            vals = line.strip().split()
            if len(vals) >= 6:
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
                colors.append([int(vals[3]), int(vals[4]), int(vals[5])])
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)

print("Loading sparse point cloud...")
points_3d, colors_3d = load_ply(PROJECT / "reconstruction/colmap_out/sparse_pc.ply")
print(f"Loaded {len(points_3d)} 3D points")

# Build feature database from training images with KNOWN poses
# For each training image, extract SIFT and associate keypoints with 3D points
# using the image's downscaled version (images_4 folder matches colmap output)
print("Building feature database...")
sift = cv2.SIFT_create(nfeatures=3000)

db_desc = []
db_pts3d = []

train_img_dir = PROJECT / "reconstruction/colmap_out/images"
train_imgs = sorted(train_img_dir.glob("*.jpg"))

# Use color matching to associate 2D keypoints with 3D point cloud points
# Build a color lookup from 3D points
print(f"Processing {len(train_imgs)} training images...")

for idx, img_path in enumerate(train_imgs[:100]):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    kps, descs = sift.detectAndCompute(img_gray, None)
    if descs is None:
        continue

    # For each keypoint, find the nearest 3D point by projecting
    # We use the frame's known c2w from transforms.json
    frame_name = img_path.name
    frame_data = None
    for frame in transforms["frames"]:
        if Path(frame["file_path"]).name == frame_name:
            frame_data = frame
            break

    if frame_data is None:
        continue

    # Get c2w matrix and invert to w2c
    c2w = np.array(frame_data["transform_matrix"], dtype=np.float64)
    # Convert nerfstudio convention back to opencv
    c2w_cv = c2w.copy()
    c2w_cv[:, 1:3] *= -1
    w2c = np.linalg.inv(c2w_cv)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Project all 3D points into this image
    pts_cam = (R @ points_3d.T).T + t
    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]
    pts_3d_valid = points_3d[valid]

    pts_proj = pts_cam[:, :2] / pts_cam[:, 2:3]
    pts_proj[:, 0] = pts_proj[:, 0] * fx + cx
    pts_proj[:, 1] = pts_proj[:, 1] * fy + cy

    in_frame = (pts_proj[:, 0] >= 0) & (pts_proj[:, 0] < w) & \
               (pts_proj[:, 1] >= 0) & (pts_proj[:, 1] < h)
    pts_proj = pts_proj[in_frame]
    pts_3d_valid = pts_3d_valid[in_frame]

    if len(pts_proj) == 0:
        continue

    # For each keypoint find nearest projected 3D point
    for kp, desc in zip(kps, descs):
        kp_pt = np.array(kp.pt)
        dists = np.linalg.norm(pts_proj - kp_pt, axis=1)
        nearest = np.argmin(dists)
        if dists[nearest] < 5.0:  # within 5 pixels
            db_desc.append(desc)
            db_pts3d.append(pts_3d_valid[nearest])

    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx+1}/{min(100, len(train_imgs))} images, {len(db_desc)} correspondences so far")

db_desc = np.array(db_desc, dtype=np.float32)
db_pts3d = np.array(db_pts3d, dtype=np.float32)
print(f"Database: {len(db_desc)} 2D-3D correspondences")

# Match test images against database
FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(
    dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
    dict(checks=100)
)

results = {}

for img_path in sorted(TEST_IMAGES.glob("*.jpg")):
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        continue

    kps, descs = sift.detectAndCompute(img_gray, None)
    if descs is None or len(descs) < 10:
        print(f"[WARN] {img_path.name}: too few features")
        continue

    matches = flann.knnMatch(descs.astype(np.float32), db_desc, k=2)

    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    print(f"{img_path.name}: {len(good)} good matches")

    if len(good) < 10:
        print(f"[WARN] {img_path.name}: too few matches")
        continue

    pts_2d = np.array([kps[m.queryIdx].pt for m in good], dtype=np.float32)
    pts_3d = np.array([db_pts3d[m.trainIdx] for m in good], dtype=np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, camera_matrix, dist_coeffs,
        reprojectionError=8.0, confidence=0.99,
        iterationsCount=1000
    )

    if not success or inliers is None or len(inliers) < 6:
        print(f"[WARN] {img_path.name}: PnP failed (inliers={len(inliers) if inliers is not None else 0})")
        continue

    R, _ = cv2.Rodrigues(rvec)
    T = tvec.flatten()

    results[img_path.name] = {
        "R": R.tolist(),
        "t": T.tolist(),
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "num_matches": len(good),
        "num_inliers": int(len(inliers))
    }
    print(f"[OK] {img_path.name}: {len(inliers)} inliers, t={T.round(4)}")

with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved SIFT+PnP results for {len(results)}/10 images to {OUTPUT}")
