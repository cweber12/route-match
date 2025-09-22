# app/transform/linear_interpolate.py
# -------------------------------------------------------------
# Linear interpolation of pose landmarks between two frames
# to create smooth transitions between processed frames. 
# -------------------------------------------------------------

import numpy as np

def linear_interpolate_pose(pose1, pose2, alpha):   
    shared_keys = set(pose1.keys()) & set(pose2.keys())
    result = {}
    
    for k in shared_keys:
        v1, v2 = pose1[k], pose2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            # Recursively handle nested dictionaries (e.g., body parts with sub-landmarks)
            result[k] = linear_interpolate_pose(v1, v2, alpha)
        else:
            # Vectorized calculation - significantly faster than individual operations
            # Convert to NumPy arrays once and perform batch mathematical operations
            v1_arr = np.asarray(v1, dtype=np.float32)
            v2_arr = np.asarray(v2, dtype=np.float32)
            result[k] = (1 - alpha) * v1_arr + alpha * v2_arr   
    return result