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


# This function matches features between two sets of descriptors using the BFMatcher.
# It filters matches based on a distance ratio and a distance threshold.
def match_features(
    desc1,
    desc2,
    ratio_thresh,
    distance_thresh,
    top_n,
    min_required_matches,
    prev_query_indices=None,
    min_shared_matches=0,
    debug=False
):
    

    # Set default values for None parameters
    if ratio_thresh is None:
        ratio_thresh = 0.75
    if distance_thresh is None:
        distance_thresh = 500.0
    if top_n is not None:
        top_n = int(top_n)

    # Force garbage collection before creating matcher
    import gc
    import time
    import sys

    # Clear any numpy/OpenCV caches
    gc.collect()

    # Clear OpenCV state
    cv2.setUseOptimized(False)
    cv2.setUseOptimized(True)

    # Add randomness to prevent any caching
    timestamp = int(time.time() * 1000000) % 1000000

    # Create a fresh BFMatcher for each call with unique seed
    cv2.setRNGSeed(timestamp)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    if debug:
        print(f"Created fresh BFMatcher with seed: {timestamp}")

    # Defensive: ensure descriptors are valid
    if desc1 is None or desc2 is None:
        if debug:
            print("[match_features] One or both descriptors are None, returning empty matches")
        return []
    if not hasattr(desc1, 'shape') or not hasattr(desc2, 'shape'):
        if debug:
            print("[match_features] One or both descriptors missing shape attribute, returning empty matches")
        return []
    if len(desc1) == 0 or len(desc2) == 0:
        if debug:
            print("[match_features] One or both descriptors are empty, returning empty matches")
        return []

    # Additional check for descriptor types and shapes
    if desc1.dtype != np.float32:
        desc1 = desc1.astype(np.float32)
    if desc2.dtype != np.float32:
        desc2 = desc2.astype(np.float32)

    if debug:
        print(f"Matching descriptors: desc1 shape {desc1.shape}, desc2 shape {desc2.shape}")

    try:
        matches = bf.knnMatch(desc1, desc2, k=2)
    except Exception as e:
        if debug:
            print(f"Error in knnMatch: {e}")
        return []
    
    # Filter out matches where we don't have 2 neighbors (shouldn't happen with proper descriptors)
    valid_matches = [m for m in matches if len(m) == 2]
    
    if debug:
        print(f"Got {len(matches)} raw matches, {len(valid_matches)} valid matches")
    
    # Debug: print first 10 match distances before filtering
    if debug and len(valid_matches) > 0:
        print("First 10 match distances (m.distance, n.distance):")
        for i, (m, n) in enumerate(valid_matches[:10]):
            print(f"  Match {i}: m.distance={m.distance:.4f}, n.distance={n.distance:.4f}")

    good = []
    for m, n in valid_matches:
        # More stringent ratio test for better quality matches
        if m.distance < ratio_thresh * n.distance and m.distance < distance_thresh:
            good.append(m)

    # Sort by distance and keep only the best matches
    good = sorted(good, key=lambda x: x.distance)
    
    # Additional filtering: Remove matches that are too far from the median distance
    if len(good) > min_required_matches:
        distances = [m.distance for m in good]
        median_distance = np.median(distances)
        std_distance = np.std(distances)
        
        # Filter out outliers (matches that are too far from median)
        filtered_good = []
        for m in good:
            if abs(m.distance - median_distance) <= 2 * std_distance:
                filtered_good.append(m)
        
        if len(filtered_good) >= min_required_matches:
            good = filtered_good
            if debug:
                print(f"Filtered outliers: {len(good)} matches remain after outlier removal")
    
    # Keep only top-N matches
    good = good[:top_n]

    if debug:
        print(f"Found {len(valid_matches)} valid matches, {len(good)} good matches")

    if len(good) < min_required_matches:
        if debug:
            print(f"Not enough stable matches ({len(good)} < {min_required_matches}), skipping this frame.")
        return []

    # Optional: Enforce shared matches with previous frame
    if prev_query_indices is not None:
        print(f"Previous query indices: {prev_query_indices}")
        shared = [m for m in good if m.queryIdx in prev_query_indices]
        if debug:
            print(f"✔ Shared matches with previous frame: {len(shared)}")
        if len(shared) < min_shared_matches:
            if debug:
                print(f"Only {len(shared)} shared matches (min required: {min_shared_matches}), skipping frame.")
            return []

    # Clean up matcher
    del bf
    gc.collect()
    
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

