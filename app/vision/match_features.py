import numpy as np
import cv2
import heapq

def match_features(
    desc1,
    desc2,
    ratio_thresh=0.75,
    distance_thresh=500.0,
    top_n=100,
    min_required_matches=10,
    prev_query_indices=None,
    min_shared_matches=0,
    debug=False
):
    # Validate descriptors
    if desc1 is None or desc2 is None:
        if debug: print("[match_features] One or both descriptors are None.")
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        if debug: print("[match_features] One or both descriptors are too small for knnMatch.")
        return []
    if not hasattr(desc1, 'shape') or not hasattr(desc2, 'shape'):
        if debug: print("[match_features] Missing shape attribute in descriptors.")
        return []

    # Ensure dtype
    if desc1.dtype != np.float32:
        desc1 = desc1.astype(np.float32)
    if desc2.dtype != np.float32:
        desc2 = desc2.astype(np.float32)

    if debug:
        print(f"[match_features] Matching descriptors: desc1 shape {desc1.shape}, desc2 shape {desc2.shape}")

    # Match using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    try:
        raw_matches = bf.knnMatch(desc1, desc2, k=2)
    except Exception as e:
        if debug: print(f"[match_features] Error in knnMatch: {e}")
        return []

    valid_matches = [m for m in raw_matches if len(m) == 2]
    if debug:
        print(f"[match_features] Got {len(raw_matches)} raw matches, {len(valid_matches)} valid matches")

    # Apply Lowe's ratio test and distance threshold
    good = []
    for m, n in valid_matches:
        if m.distance < ratio_thresh * n.distance and m.distance < distance_thresh:
            good.append(m)

    if len(good) == 0:
        if debug: print("[match_features] No good matches passed ratio and distance threshold.")
        return []

    # Optional: filter out outliers based on median + std (if large set)
    if len(good) > 100:
        distances = np.array([m.distance for m in good])
        median_distance = np.median(distances)
        std_distance = np.std(distances)
        good = [m for m in good if abs(m.distance - median_distance) <= 2 * std_distance]
        if debug: print(f"[match_features] Filtered outliers: {len(good)} remain after median/std filtering")

    if len(good) < min_required_matches:
        if debug: print(f"[match_features] Not enough matches ({len(good)} < {min_required_matches})")
        return []

    # Keep only top-N matches
    good = heapq.nsmallest(top_n, good, key=lambda x: x.distance)

    # Optional: enforce shared queryIdx matches with previous frame
    if prev_query_indices is not None:
        shared = [m for m in good if m.queryIdx in prev_query_indices]
        if debug:
            print(f"[match_features] Shared matches: {len(shared)} (required: {min_shared_matches})")
        if len(shared) < min_shared_matches:
            return []

    return good