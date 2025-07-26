import numpy as np
import cv2


def get_transformation_info(T):
    """Extract readable info from transformation matrix"""
    if T is None:
        return "No transformation"
    
    # Extract rotation and scaling components
    scale_x = np.sqrt(T[0, 0]**2 + T[0, 1]**2)
    scale_y = np.sqrt(T[1, 0]**2 + T[1, 1]**2)
    
    # Calculate rotation angle (in radians)
    rotation_angle = np.arctan2(T[1, 0], T[0, 0])
    rotation_degrees = np.degrees(rotation_angle)
    
    # Translation components
    trans_x = T[0, 2]
    trans_y = T[1, 2]
    
    return f"Scale: X={scale_x:.3f}, Y={scale_y:.3f} | Rotation: {rotation_degrees:.1f}° | Translation: ({trans_x:.1f}, {trans_y:.1f})"

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

    if debug:
        print(f"Computing affine transform with {len(matches)} matches")
        print(f"  Source points sample: {src_pts[:3]}")
        print(f"  Destination points sample: {dst_pts[:3]}")

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

    if debug:
        print(f"Computed transformation matrix: {T}")
        print(f"Inliers: {np.sum(inliers) if inliers is not None else 'None'}")

    # Validate transformation to prevent extreme rotations/scaling
    def validate_transformation(transform):
        """Check if transformation is reasonable and not extreme"""
        if transform is None:
            return False
        
        # Extract rotation and scaling components
        scale_x = np.sqrt(transform[0, 0]**2 + transform[0, 1]**2)
        scale_y = np.sqrt(transform[1, 0]**2 + transform[1, 1]**2)
        
        # Calculate rotation angle (in radians)
        rotation_angle = np.arctan2(transform[1, 0], transform[0, 0])
        rotation_degrees = np.degrees(rotation_angle)
        
        # Check for reasonable scaling (between 0.3 and 3.0)
        if scale_x < 0.3 or scale_x > 3.0 or scale_y < 0.3 or scale_y > 3.0:
            if debug:
                print(f"Extreme scaling detected: X={scale_x:.3f}, Y={scale_y:.3f}")
            return False
        
        # Check for reasonable rotation (less than 15 degrees for climbing images)
        if abs(rotation_degrees) > 15:
            if debug:
                print(f"Extreme rotation detected: {rotation_degrees:.1f} degrees (limit: 15°)")
            return False
        
        return True
    
    # Validate the computed transformation
    if not validate_transformation(T):
        if debug:
            print("Transformation validation failed, using previous transform")
        return prev_T
    
    # Only apply smoothing if alpha > 0 and we have a previous transform
    if prev_T is not None and alpha > 0:
        # Smooth the transform
        T_smoothed = alpha * prev_T + (1 - alpha) * T
        if debug:
            print(f"Applied smoothing with alpha={alpha}")
            print(f"  Smoothed transform:")
            print(f"  {T_smoothed}")
        
        # Validate smoothed transformation as well
        if not validate_transformation(T_smoothed):
            if debug:
                print("Smoothed transformation validation failed, using raw transform")
            return T
        
        return T_smoothed
    else:
        if debug:
            print("No smoothing applied (alpha=0 or no previous transform)")
        return T

