# app/pipelines/video_tripod.py
# -------------------------------------------------------------
# Generate video from a single reference image and pose landmarks
# using SIFT feature matching to compute a single affine transformation.
# -------------------------------------------------------------

import os
import cv2
import numpy as np
from pathlib import Path
import subprocess
import json
import shutil
import gc
import sys
import time
import logging
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading
import base64

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


#from .video_writer import write_pose_video
from ..vision.detect_img_sift import detect_sift
from ..vision.match_features import match_features
from ..transform.draw_points import apply_transform, draw_pose
from ..jobs.job_manager import update_job_progress
from ..transform.affine_transform import compute_affine_transform 
from ..transform.linear_interpolate import linear_interpolate_pose
from ..performance.performance_monitor import performance_monitor

VIDEO_OUT_DIR = os.path.join("temp_uploads", "pose_feature_data", "output_video")
POSE_JSON = os.path.join("static", "pose_feature_data", "pose_landmarks.json")
SIFT_JSON = os.path.join("static", "pose_feature_data", "sift_keypoints.json")

def save_current_frame(frame_buffer):
    out_dir = os.path.join("temp_uploads", "pose_feature_data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "current.jpg")
    success, img_bytes = cv2.imencode('.jpg', frame_buffer)
    if success:
        with open(out_path, "wb") as f:
            f.write(img_bytes.tobytes())

# Generate all video frames from the reference image and transformed poses
def generate_video_frames(ref_img, transformed, frame_keys, line_color, point_color, writer, job_id: Optional[str] = None):
    
    # Pre-allocate frame buffer to avoid memory thrashing
    h, w = ref_img.shape[:2]
    frame_buffer = np.empty((h, w, 3), dtype=np.uint8)

    logger.info(f"Starting frame generation for {len(frame_keys)} key frames")
    total_frames_written = 0
    total_frames = frame_keys[len(frame_keys)-1] - frame_keys[0] + 1
    
    # Iterate through consecutive frame pairs and generate all interpolated
    # frames for each segment in one batch, reducing overhead
    for i in range(len(frame_keys) - 1):
        f1, f2 = frame_keys[i], frame_keys[i + 1]
        pose1, pose2 = transformed[f1], transformed[f2]
        frame_gap = f2 - f1
        
        # Write the first frame of this segment
        np.copyto(frame_buffer, ref_img)
        draw_pose(frame_buffer, pose1, line_color, point_color)
        writer.write(frame_buffer)
        total_frames_written += 1
        save_current_frame(frame_buffer)
        # Update job progress at the specified frame rate (not every frame) 
        pct = int((total_frames_written / total_frames) * 100)  
        try:
            # Passes job progress percentage to job manager
            update_job_progress(job_id, pct)
        except Exception:
            pass
  
        # Batch interpolation for frame gaps
        # If there's a gap between frames (0, 15, 30, ... , 15*n), generate interpolated
        # frames for smooth animation. Pre-calculate all alpha values using NumPy's linspace
        if frame_gap > 1:
            logger.debug(f"Generating {frame_gap-1} interpolated frames between {f1} and {f2}")
            
            # Pre-calculate all interpolation factors (alpha values) at once
            # Faster than calculating j / frame_gap in a loop
            alphas = np.linspace(1/frame_gap, (frame_gap-1)/frame_gap, frame_gap-1)
            
            # Generate all interpolated frames for this segment
            for alpha in alphas:
                # Reuse the same frame buffer (no new memory allocation)
                np.copyto(frame_buffer, ref_img)
                
                # Use vectorized interpolation 
                interp_pose = linear_interpolate_pose(pose1, pose2, alpha)
                draw_pose(frame_buffer, interp_pose, line_color, point_color)
                writer.write(frame_buffer)
                total_frames_written += 1
                save_current_frame(frame_buffer)
                  
    # Write the final frame
    # Handle the last frame separately since the loop processes pairs
    np.copyto(frame_buffer, ref_img)
    draw_pose(frame_buffer, transformed[frame_keys[-1]], line_color, point_color)
    writer.write(frame_buffer)
    total_frames_written += 1
    
    logger.info(f"Successfully generated {total_frames_written} total video frames")
    
    # Explicitly delete the frame buffer to free memory immediately
    del frame_buffer

@performance_monitor

# Main function to generate video from a single image and pose landmarks
def generate_video(
    image_path,
    pose_landmarks,
    stored_keypoints_all, 
    stored_descriptors_all,
    output_video="output_video.mp4",
    sift_left=20.0,
    sift_right=20.0,
    sift_up=20.0,
    sift_down=20.0,
    line_color=(100, 255, 0),
    point_color=(0, 100, 255),
    frame_dimensions=None,
    fps=24, 
    job_id: Optional[str] = None

):

    # Clean up local storage ONCE before any S3 downloads
    from ..storage.s3.load_json_s3 import cleanup_local_storage
    cleanup_local_storage()

    # Clear any existing OpenCV state to avoid memory leaks
    gc.collect()
    cv2.setUseOptimized(True)
    cv2.setRNGSeed(0)
    # Clear any cached modules that may hold onto large data
    for module_name in [
        "app.services.detect_img_sift",
        "app.services.match_features",
        "app.services.draw_points",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]
    # Force garbage collection
    gc.collect()

    ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if ref_img is None:
        logger.error(f"Failed to load reference image from {image_path}")
        return {"error": "Image load failed"}
    ref_img = ref_img.copy()
    if frame_dimensions is not None:
        if isinstance(frame_dimensions, tuple) and len(frame_dimensions) == 2:
            stored_width, stored_height = frame_dimensions
            stored_resolution = stored_width * stored_height
            ref_resolution = ref_img.shape[1] * ref_img.shape[0]
            if ref_resolution > stored_resolution:
                scale_factor = stored_resolution / ref_resolution
                ref_img = cv2.resize(ref_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    # Ensure we're using absolute paths
    abs_video_out_dir = os.path.abspath(VIDEO_OUT_DIR)
    os.makedirs(abs_video_out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(abs_video_out_dir, output_video))
    
    logger.info(f"Video output directory: {abs_video_out_dir}")
    logger.info(f"Video output path: {out_path}")
    
    # Clean up existing MP4 files
    for f in os.listdir(abs_video_out_dir):
        if f.endswith(".mp4"):
            old_file = os.path.join(abs_video_out_dir, f)
            os.remove(old_file)
            logger.debug(f"Removed old video file: {old_file}")

    # Define SIFT detection bounding box based on percentage parameters
    h, w = ref_img.shape[:2]
    x1 = int(sift_left / 100 * w)
    y1 = int(sift_up / 100 * h)
    x2 = int(w - (sift_right / 100) * w)
    y2 = int(h - (sift_down / 100) * h)
    bbox = (x1, y1, x2, y2)

    # Initialize SIFT detector with a unique number of features to avoid caching issues
    timestamp = int(time.time() * 1000) % 1000
    sift_config = {
        "nfeatures": 2000 + timestamp,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6,
    }

    # Create SIFT detector
    detector = cv2.SIFT_create(
        nfeatures=sift_config["nfeatures"],
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )

    # Detect SIFT features in the reference image within the bounding box
    ref_kp, ref_desc = detect_sift(
        ref_img, 
        sift_config=sift_config, 
        bbox=bbox, 
        detector=detector
    )
    
    if not ref_kp or ref_desc is None or len(ref_kp) == 0:
        logger.error("SIFT feature detection failed - unable to proceed with video generation")
        logger.error(f"Image dimensions: {h}x{w}, bbox: {bbox}")
        return {"error": "SIFT failure"}

    frame_keys = sorted([int(k) for k in pose_landmarks.keys()])
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        logger.error(f"Failed to initialize video writer for {out_path}")
        return {"error": "Video writer failed"}

    kp = stored_keypoints_all[0]
    desc = stored_descriptors_all[0]
    
    logger.info(f"Static mode: Stored SIFT features - {len(kp)} keypoints, descriptor shape: {desc.shape if desc is not None else 'None'}")
    logger.info(f"Reference SIFT features - {len(ref_kp)} keypoints, descriptor shape: {ref_desc.shape if ref_desc is not None else 'None'}")
    
    matches = match_features(
        desc1=desc,
        desc2=ref_desc,
        ratio_thresh=0.6,  # Even more strict for better matches
        distance_thresh=300,  # Tighter distance threshold
        top_n=500,  # Consider more matches
        min_required_matches=3,  # Lower minimum
        prev_query_indices=None,
        min_shared_matches=0,
        debug=True  # Enable debug output
    )
    
    logger.info(f"SIFT matching result: {len(matches) if matches else 0} matches found")
    
    if not matches:
        logger.warning("No matches found - trying more lenient parameters")
        # Try again with more lenient parameters
        matches = match_features(
            desc1=desc,
            desc2=ref_desc,
            ratio_thresh=0.8,  # More lenient
            distance_thresh=600,  # Higher distance
            top_n=200,  # Fewer matches for quality
            min_required_matches=3,  # Minimum possible
            prev_query_indices=None,
            min_shared_matches=0,
            debug=True
        )
        logger.info(f"Second attempt: {len(matches) if matches else 0} matches found")
    
    if not matches:
        logger.error("Still no matches after lenient retry - this indicates fundamental incompatibility between images")
        writer.release()
        return "NO_MATCHES"
        
    # Try to filter matches for better geometric consistency
    if len(matches) > 10:
        # Sort by distance and take the best ones
        matches = sorted(matches, key=lambda x: x.distance)[:15]
        logger.info(f"Filtered to top {len(matches)} matches by distance")
    # Try multiple approaches for robust transformation
    T = None
    transform_attempts = [
        # Attempt 1: Conservative RANSAC with stricter validation
        {
            "ransac_thresh": 1.0,
            "max_iters": 5000,
            "confidence": 0.99,
            "alpha": 0.0,
            "min_required_matches": max(3, min(len(matches)//2, 8)),
            "description": "Conservative"
        },
        # Attempt 2: More lenient RANSAC
        {
            "ransac_thresh": 2.5,
            "max_iters": 8000,
            "confidence": 0.95,
            "alpha": 0.0,
            "min_required_matches": 3,
            "description": "Lenient"
        },
        # Attempt 3: Very permissive for difficult cases
        {
            "ransac_thresh": 5.0,
            "max_iters": 10000,
            "confidence": 0.85,
            "alpha": 0.0,
            "min_required_matches": 3,
            "description": "Permissive"
        }
    ]
    
    # Try each transformation attempt until one succeeds
    for i, params in enumerate(transform_attempts):
        logger.info(f"Transform attempt {i+1} ({params['description']}): using {len(matches)} matches")
        # Compute the affine transformation matrix
        T_attempt = compute_affine_transform(
            kp, ref_kp, matches,
            prev_T=None,
            ransac_thresh=params["ransac_thresh"],
            max_iters=params["max_iters"],
            confidence=params["confidence"],
            alpha=params["alpha"],
            min_required_matches=params["min_required_matches"],
            debug=True
        )
        
        # Validate the computed transformation
        if T_attempt is not None:
            # Validate the transformation
            scale_x = np.sqrt(T_attempt[0, 0] ** 2 + T_attempt[0, 1] ** 2)
            scale_y = np.sqrt(T_attempt[1, 0] ** 2 + T_attempt[1, 1] ** 2)
            
            # More lenient validation criteria
            if (0.1 <= scale_x <= 10.0 and 0.1 <= scale_y <= 10.0):
                logger.info(f"Transform attempt {i+1} succeeded: scale=({scale_x:.3f}, {scale_y:.3f})")
                T = T_attempt
                break
            else:
                logger.warning(f"Transform attempt {i+1} failed validation: scale=({scale_x:.3f}, {scale_y:.3f})")
        else:
            logger.warning(f"Transform attempt {i+1} returned None")
    
    # If all attempts failed, log error
    if T is None:
        logger.error("All transformation attempts failed")
        writer.release()
        return "NO_TRANSFORM"

    logger.info("Successfully computed transformation matrix")

    # Pre-transform all pose landmarks using the computed transformation matrix
    # This is done once upfront to avoid repeated transform calculations
    transformed = {
        frame: apply_transform(T, pose_landmarks[frame])
        for frame in frame_keys
    }
    
    # Generate video frames
    generate_video_frames(
        ref_img, transformed, frame_keys, line_color, point_color, writer, job_id=job_id
    )
    
    writer.release()
    
    # Force filesystem sync to ensure file is fully written
    time.sleep(0.5)  # Give the filesystem time to flush
    
    logger.info(f"Finished writing video to {out_path}")
    
    # Verify the file was actually created and get its size
    if os.path.exists(out_path):
        file_size = os.path.getsize(out_path)
        logger.info(f"Video file verification: {out_path} exists, size: {file_size} bytes")
        if file_size == 0:
            logger.error(f"Video file is empty: {out_path}")
            return "EMPTY_VIDEO"
        
        # Double-check with a small delay
        time.sleep(0.2)
        if os.path.exists(out_path):
            final_size = os.path.getsize(out_path)
            logger.info(f"Final verification: file still exists, size: {final_size} bytes")
        else:
            logger.error(f"File disappeared during verification: {out_path}")
            return "FILE_DISAPPEARED"
            
    else:
        logger.error(f"Video file was not created at expected path: {out_path}")
        # List what files actually exist in the directory
        if os.path.exists(abs_video_out_dir):
            existing_files = os.listdir(abs_video_out_dir)
            logger.error(f"Files in output directory: {existing_files}")
        else:
            logger.error(f"Output directory doesn't exist: {abs_video_out_dir}")
        return "FILE_NOT_CREATED"
    
    return "SUCCESS"
