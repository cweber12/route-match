import numpy as np
import cv2

# This function matches features between two sets of descriptors using the BFMatcher.
# It filters matches based on a distance ratio and a distance threshold.
def match_features(
    desc1,
    desc2,
    ratio_thresh,
    distance_thresh=300,
    top_n=150,
    min_required_matches=10,
    prev_query_indices=None,
    min_shared_matches=5,
    debug=False
):

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance and m.distance < distance_thresh:
            good.append(m)

    # Sort and keep only top-N matches
    good = sorted(good, key=lambda x: x.distance)[:top_n]

    if debug:
        print(f"Found {len(matches)} raw matches, {len(good)} good matches")

    if len(good) < min_required_matches:
        if debug:
            print("Not enough stable matches, skipping this frame.")
        return []

    # Optional: Enforce shared matches with previous frame
    if prev_query_indices is not None:
        shared = [m for m in good if m.queryIdx in prev_query_indices]
        if debug:
            print(f"✔ Shared matches with previous frame: {len(shared)}")
        if len(shared) < min_shared_matches:
            if debug:
                print(f"Only {len(shared)} shared matches (min required: {min_shared_matches}), skipping frame.")
            return []

    return good

def compute_affine_transform(
    kp1,
    kp2,
    matches,
    prev_T=None,
    ransac_thresh=1.0,
    max_iters=5000,
    confidence=0.999,
    alpha=0.9,
    min_required_matches=3,
    debug=False
):
    
    if len(matches) < min_required_matches:
        if debug:
            print(f"Not enough matches ({len(matches)}), using previous transform.")
        return prev_T

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    T, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=max_iters,
        confidence=confidence
    )

    if T is None:
        if debug:
            print("Transform estimation failed, using previous.")
        return prev_T

    if prev_T is not None:
        # Smooth the transform
        T = alpha * prev_T + (1 - alpha) * T

    return T

