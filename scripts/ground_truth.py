import cv2
import numpy as np
import json
import os
from pathlib import Path

# ArUco board parameters
DICT_TYPE = cv2.aruco.DICT_4X4_50
MARKERS_X = 5
MARKERS_Y = 7
MARKER_SIZE = 0.03      # 3 cm in meters
MARKER_GAP = 0.006      # 0.6 cm in meters

# Paths
PROJECT = Path("/mnt/c/Users/i3pur1la/Desktop/SpaceChallenges")
TEST_IMAGES = PROJECT / "data/test/images"
OUTPUT = PROJECT / "data/test/ground_truth.json"

# Load camera intrinsics from COLMAP transforms.json
transforms_path = PROJECT / "reconstruction/colmap_out/transforms.json"
with open(transforms_path) as f:
    transforms = json.load(f)

fx = transforms["fl_x"]
fy = transforms["fl_y"]
cx = transforms["cx"]
cy = transforms["cy"]
w  = transforms["w"]
h  = transforms["h"]

k1 = transforms.get("k1", 0)
k2 = transforms.get("k2", 0)
p1 = transforms.get("p1", 0)
p2 = transforms.get("p2", 0)

camera_matrix = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float64)
dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64)

# Create ArUco board
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
board = cv2.aruco.GridBoard(
    (MARKERS_X, MARKERS_Y),
    MARKER_SIZE,
    MARKER_GAP,
    aruco_dict
)

detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

results = {}

for img_path in sorted(TEST_IMAGES.glob("*.jpg")):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) < 3:
        print(f"[WARN] {img_path.name}: only {0 if ids is None else len(ids)} markers detected — skipping")
        continue

    obj_points, img_points = board.matchImagePoints(corners, ids)

    if obj_points is None or len(obj_points) < 4:
        print(f"[WARN] {img_path.name}: not enough matched points — skipping")
        continue

    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print(f"[WARN] {img_path.name}: PnP failed — skipping")
        continue

    R, _ = cv2.Rodrigues(rvec)
    T = tvec.flatten()

    results[img_path.name] = {
        "R": R.tolist(),
        "t": T.tolist(),
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "num_markers": int(len(ids))
    }
    print(f"[OK] {img_path.name}: {len(ids)} markers, t={T.round(4)}")

with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved ground truth for {len(results)}/10 images to {OUTPUT}")
