# app/transform/affine_transform.py
# -------------------------------------------------------------
# Compute and validate affine transformations between sets of keypoints
# using OpenCV, with smoothing and sanity checks to avoid extreme warps.
# Used to create affine transformation matrices from matched SIFT keypoints
# that are applied to stored pose landmarks. 
# -------------------------------------------------------------

import numpy as np
import cv2

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