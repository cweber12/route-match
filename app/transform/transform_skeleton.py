# compare_pose.py
# ---------------------------------------------------------------------------
# This file contains functions that use matched SIFT data to create a 
# transformation matrix that is applied to stored pose landmark coordinates. 
# Linear interpolation is used to fill in gaps between saved frames for 
# smoother transitions. 
# ---------------------------------------------------------------------------

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
from ..vision.match_features import match_features, compute_affine_transform
from .draw_points import apply_transform, draw_pose
from ..jobs.job_manager import update_job_progress 

VIDEO_OUT_DIR = os.path.join("temp_uploads", "pose_feature_data", "output_video")
POSE_JSON = os.path.join("static", "pose_feature_data", "pose_landmarks.json")
SIFT_JSON = os.path.join("static", "pose_feature_data", "sift_keypoints.json")

# Performance monitoring: 
# Logs execution time and estimates frames per second for video generation
# Used to compare performance of different implementations
def performance_monitor(func):

    def wrapper(*args, **kwargs):
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        logger.info(f"Performance Stats for {func.__name__}:")
        logger.info(f"  Execution time: {execution_time:.2f} seconds")
        
        # Estimate frames per second if this was frame generation
        if 'frame' in func.__name__.lower() and execution_time > 0:
            # Try to estimate frame count from args/kwargs
            frame_count = 0
            for arg in args:
                if isinstance(arg, dict) and hasattr(arg, '__len__'):
                    frame_count = len(arg)
                    break
            
            if frame_count > 0:
                fps = frame_count / execution_time
                logger.info(f"  Frame generation rate: {fps:.1f} frames/second")
                logger.info(f"  Average time per frame: {execution_time/frame_count:.3f} seconds")
        
        return result
    return wrapper

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
                
                # Use vectorized interpolation (3-5x faster than original)
                interp_pose = linear_interpolate_pose(pose1, pose2, alpha)
                draw_pose(frame_buffer, interp_pose, line_color, point_color)
                
                writer.write(frame_buffer)
                total_frames_written += 1
    
    # Write the final frame
    # Handle the last frame separately since the loop processes pairs
    np.copyto(frame_buffer, ref_img)
    draw_pose(frame_buffer, transformed[frame_keys[-1]], line_color, point_color)
    writer.write(frame_buffer)
    total_frames_written += 1
    
    logger.info(f"Successfully generated {total_frames_written} total video frames")
    
    # Explicitly delete the frame buffer to free memory immediately
    # This is especially important for large images or when processing multiple videos
    del frame_buffer

@performance_monitor

# Clears the output video from the previous run before creating a new one
def clear_output_dir():
    for file in os.listdir(VIDEO_OUT_DIR):
        file_path = os.path.join(VIDEO_OUT_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def validate_affine_transform(T):
    try:
        scale_x = np.sqrt(T[0, 0] ** 2 + T[0, 1] ** 2)
        scale_y = np.sqrt(T[1, 0] ** 2 + T[1, 1] ** 2)
        rotation_angle = np.arctan2(T[1, 0], T[0, 0]) * 180 / np.pi
        translation_x = T[0, 2]
        translation_y = T[1, 2]
        
        # Log transformation details
        logger.info(f"Transform analysis:")
        logger.info(f"  Scale: ({scale_x:.3f}, {scale_y:.3f})")
        logger.info(f"  Rotation: {rotation_angle:.1f}°")
        logger.info(f"  Translation: ({translation_x:.1f}, {translation_y:.1f})")
        
        # Warnings for unusual but acceptable transformations
        if scale_x > 5.0 or scale_y > 5.0:
            logger.warning(f"Large scaling detected: {scale_x:.2f}x, {scale_y:.2f}x")
        elif scale_x < 0.2 or scale_y < 0.2:
            logger.warning(f"Small scaling detected: {scale_x:.2f}x, {scale_y:.2f}x")
        
        if abs(rotation_angle) > 45:
            logger.warning(f"Large rotation detected: {rotation_angle:.1f}°")
        
        if abs(translation_x) > 1000 or abs(translation_y) > 1000:
            logger.warning(f"Large translation detected: ({translation_x:.1f}, {translation_y:.1f})")
        
        # Only log success if no major warnings
        if (0.2 <= scale_x <= 5.0 and 0.2 <= scale_y <= 5.0 and 
            abs(rotation_angle) <= 45 and 
            abs(translation_x) <= 1000 and abs(translation_y) <= 1000):
            logger.info("Transform validation passed - good quality transformation")
        else:
            logger.info("Transform validation completed with warnings - proceed with caution")
        
    except Exception as e:
        logger.error(f"Error validating affine matrix: {e}")

@performance_monitor

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
    # set the fps for the output video
    fps=24, 
    job_id: Optional[str] = None

):

    # Clean up local storage ONCE before any S3 downloads
    from ..storage.s3.load_json_s3 import cleanup_local_storage
    cleanup_local_storage()
    logger.info("Starting optimized video generation with enhanced performance monitoring")

    gc.collect()
    cv2.setUseOptimized(True)
    cv2.setRNGSeed(0)

    for module_name in [
        "app.services.detect_img_sift",
        "app.services.match_features",
        "app.services.draw_points",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

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

    h, w = ref_img.shape[:2]
    x1 = int(sift_left / 100 * w)
    y1 = int(sift_up / 100 * h)
    x2 = int(w - (sift_right / 100) * w)
    y2 = int(h - (sift_down / 100) * h)
    bbox = (x1, y1, x2, y2)

    timestamp = int(time.time() * 1000) % 1000
    sift_config = {
        "nfeatures": 2000 + timestamp,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6,
    }

    detector = cv2.SIFT_create(
        nfeatures=sift_config["nfeatures"],
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )

    ref_kp, ref_desc = detect_sift(
        ref_img, 
        sift_config=sift_config, 
        bbox=bbox, 
        detector=detector
    )
    
    logger.info(f"Reference image SIFT detection: {len(ref_kp) if ref_kp else 0} keypoints")
    logger.info(f"Reference descriptor shape: {ref_desc.shape if ref_desc is not None else 'None'}")
    logger.info(f"SIFT bbox applied: {bbox} (left: {sift_left}%, right: {sift_right}%, up: {sift_up}%, down: {sift_down}%)")
    
    if not ref_kp or ref_desc is None or len(ref_kp) == 0:
        logger.error("SIFT feature detection failed - unable to proceed with video generation")
        logger.error(f"Image dimensions: {h}x{w}, bbox: {bbox}")
        return {"error": "SIFT failure"}

    frame_keys = sorted([int(k) for k in pose_landmarks.keys()])
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        logger.error(f"Failed to initialize video writer for {out_path}")
        return {"error": "Video writer failed"}

    static_mode = len(stored_keypoints_all) == 1
    if static_mode:
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
            # Try again with very lenient parameters but fewer matches
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
        
        for i, params in enumerate(transform_attempts):
            logger.info(f"Transform attempt {i+1} ({params['description']}): using {len(matches)} matches")
            
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
        
        if T is None:
            logger.error("All transformation attempts failed")
            writer.release()
            return "NO_TRANSFORM"

        logger.info("Successfully computed transformation matrix")
        validate_affine_transform(T)

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

    writer.release()
    return

@performance_monitor
def generate_video_multiframe(
    image_path,
    pose_landmarks,
    sift_data_all,
    output_video="output_video.mp4",
    sift_left=20.0,
    sift_right=20.0,
    sift_up=20.0,
    sift_down=20.0,
    line_color=(100, 255, 0),
    point_color=(0, 100, 255),
    frame_dimensions=None,
    fps=24
):
    """
    Generate video with frame-by-frame SIFT matching for dynamic sequences.
    Each frame gets its own transformation matrix based on SIFT matching.
    """
    logger.info("Starting multi-frame video generation")
    
    # Initial setup (same as single-frame)
    from ..storage.s3.load_json_s3 import cleanup_local_storage
    cleanup_local_storage()
    
    gc.collect()
    ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if ref_img is None:
        logger.error(f"Failed to load reference image from {image_path}")
        return {"error": "Image load failed"}
    
    ref_img = ref_img.copy()
    h, w = ref_img.shape[:2]
    
    # Setup output
    os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
    out_path = os.path.join(VIDEO_OUT_DIR, output_video)
    for f in os.listdir(VIDEO_OUT_DIR):
        if f.endswith(".mp4"):
            os.remove(os.path.join(VIDEO_OUT_DIR, f))
    
    # SIFT detection on reference image
    x1 = int(sift_left / 100 * w)
    y1 = int(sift_up / 100 * h)
    x2 = int(w - (sift_right / 100) * w)
    y2 = int(h - (sift_down / 100) * h)
    bbox = (x1, y1, x2, y2)
    
    timestamp = int(time.time() * 1000) % 1000
    sift_config = {
        "nfeatures": 2000 + timestamp,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6,
    }
    
    detector = cv2.SIFT_create(
        nfeatures=sift_config["nfeatures"],
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )
    
    ref_kp, ref_desc = detect_sift(
        ref_img, 
        sift_config=sift_config, 
        bbox=bbox, 
        detector=detector
    )
    
    if not ref_kp or ref_desc is None or len(ref_kp) == 0:
        logger.error("SIFT feature detection failed on reference image")
        return {"error": "SIFT failure"}
    
    # Collect all multi-frame SIFT data
    frame_sift_data = {}
    for sift_entry in sift_data_all:
        if sift_entry["is_multi_frame"]:
            sift_frames = sift_entry["data"]
            for frame_num, frame_data in sift_frames.items():
                if frame_num not in frame_sift_data:
                    frame_sift_data[frame_num] = []
                frame_sift_data[frame_num].append(frame_data)
    
    logger.info(f"Found SIFT data for {len(frame_sift_data)} frames: {sorted(frame_sift_data.keys())}")
    
    # Compute transformation matrix for each frame
    frame_transforms = {}
    successful_frames = []
    
    prev_successful_T = None
    for frame_num in sorted(frame_sift_data.keys()):
        logger.debug(f"Computing transform for frame {frame_num}")
        
        # Combine SIFT data from all sources for this frame
        combined_kps = []
        combined_descs = []
        
        for frame_data in frame_sift_data[frame_num]:
            combined_kps.extend(frame_data["keypoints"])
            if len(combined_descs) == 0:
                combined_descs = frame_data["descriptors"]
            else:
                combined_descs = np.vstack([combined_descs, frame_data["descriptors"]])
        
        if len(combined_kps) == 0 or combined_descs.size == 0:
            logger.warning(f"No SIFT features for frame {frame_num}")
            continue
        
        # Match features with reference image
        matches = match_features(
            desc1=combined_descs,
            desc2=ref_desc,
            ratio_thresh=0.75,  # More lenient for multi-frame
            distance_thresh=500,  # Increased distance threshold
            top_n=300,  # More matches to consider
            min_required_matches=5,  # Reduced minimum requirement
            debug=False
        )
        
        if not matches:
            logger.warning(f"No matches found for frame {frame_num} - trying more lenient parameters")
            # Try again with very lenient parameters
            matches = match_features(
                desc1=combined_descs,
                desc2=ref_desc,
                ratio_thresh=0.9,
                distance_thresh=800,
                top_n=500,
                min_required_matches=3,
                debug=False
            )
        
        if not matches:
            logger.warning(f"Still no matches found for frame {frame_num}")
            continue
        
        # Compute transformation matrix with more lenient parameters
        T = compute_affine_transform(
            combined_kps, ref_kp, matches,
            prev_T=None,
            ransac_thresh=2.0,  # More lenient RANSAC
            max_iters=10000,  # More iterations
            confidence=0.95,  # Lower confidence for more tolerance
            alpha=0.0,
            min_required_matches=3,
            debug=False
        )
        
        if T is None:
            logger.warning(f"First transform attempt failed for frame {frame_num} - trying more lenient RANSAC")
            # Try again with very lenient parameters
            T = compute_affine_transform(
                combined_kps, ref_kp, matches,
                prev_T=None,
                ransac_thresh=5.0,  # Very lenient
                max_iters=15000,
                confidence=0.8,
                alpha=0.0,
                min_required_matches=3,
                debug=False
            )
        
        if T is not None:
            # Validate the transformation before accepting it
            scale_x = np.sqrt(T[0, 0] ** 2 + T[0, 1] ** 2)
            scale_y = np.sqrt(T[1, 0] ** 2 + T[1, 1] ** 2)
            used_force = False

            # Accept transformations with reasonable scaling (more lenient than single-frame)
            if 0.1 <= scale_x <= 10.0 and 0.1 <= scale_y <= 10.0:
                T_use = T
                logger.debug(f"Successfully computed transform for frame {frame_num} (scale: {scale_x:.2f}x, {scale_y:.2f}x)")
            else:
                # Try to salvage by interpolating with previous successful transform if available
                if prev_successful_T is not None:
                    try:
                        T_interp = 0.5 * prev_successful_T + 0.5 * T
                        sx = np.sqrt(T_interp[0, 0] ** 2 + T_interp[0, 1] ** 2)
                        sy = np.sqrt(T_interp[1, 0] ** 2 + T_interp[1, 1] ** 2)
                        if 0.1 <= sx <= 10.0 and 0.1 <= sy <= 10.0:
                            T_use = T_interp
                            logger.info(f"Using interpolated transform for frame {frame_num} to avoid extreme scaling (sx={sx:.2f}, sy={sy:.2f})")
                        else:
                            # mark for forced apply (clamping) later
                            T_use = T
                            used_force = True
                            logger.warning(f"Interpolated transform still extreme (sx={sx:.2f}, sy={sy:.2f}), will force-apply with clamping")
                    except Exception:
                        T_use = T
                        used_force = True
                        logger.exception("Interpolation with previous transform failed; will force-apply")
                else:
                    # No previous transform to interpolate with; force-apply computed transform
                    T_use = T
                    used_force = True
                    logger.warning(f"Rejected transform for frame {frame_num} due to extreme scaling: {scale_x:.2f}x, {scale_y:.2f}x; will force-apply")

            # Store the transform and flag
            frame_transforms[frame_num] = {"T": T_use, "force": bool(used_force)}
            successful_frames.append(frame_num)
            prev_successful_T = np.asarray(frame_transforms[frame_num]["T"], dtype=np.float64).reshape(2, 3)
        else:
            logger.warning(f"Failed to compute transform for frame {frame_num}")
    
    if len(successful_frames) == 0:
        logger.error("No successful transformations computed")
        return "NO_TRANSFORM"
    
    logger.info(f"Successfully computed transforms for {len(successful_frames)} frames")
    
    # Generate video with frame-specific transformations
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        logger.error(f"Failed to initialize video writer for {out_path}")
        return {"error": "Video writer failed"}
    
    try:
        generate_multiframe_video_frames(
            ref_img, pose_landmarks, frame_transforms, 
            successful_frames, line_color, point_color, writer
        )
    finally:
        writer.release()
    
    logger.info(f"Finished writing multi-frame video to {out_path}")
    return "SUCCESS"

def generate_multiframe_video_frames(
    ref_img, pose_landmarks, frame_transforms, 
    successful_frames, line_color, point_color, writer
):

    # Helper: interpolate two 2x3 affine matrices element-wise
    def _interpolate_affine(T1, T2, alpha: float):
        T1a = np.asarray(T1, dtype=np.float64).reshape(2, 3)
        T2a = np.asarray(T2, dtype=np.float64).reshape(2, 3)
        return (1.0 - alpha) * T1a + alpha * T2a

    h, w = ref_img.shape[:2]
    frame_buffer = np.empty((h, w, 3), dtype=np.uint8)

    # Normalize pose frame keys to ints
    pose_frame_keys = sorted([int(k) for k in pose_landmarks.keys()])
    logger.info(f"Pose frames: {pose_frame_keys[:10]}...{pose_frame_keys[-10:] if len(pose_frame_keys) > 10 else ''}")

    # Normalize transforms: support both legacy T values and new dict {"T":..., "force":bool}
    transforms_np = {}
    transforms_force = {}
    for k, v in frame_transforms.items():
        try:
            ki = int(k)
        except Exception:
            ki = k
        if isinstance(v, dict) and "T" in v:
            transforms_np[int(ki)] = np.asarray(v["T"], dtype=np.float64).reshape(2, 3)
            transforms_force[int(ki)] = bool(v.get("force", False))
        else:
            transforms_np[int(ki)] = np.asarray(v, dtype=np.float64).reshape(2, 3)
            transforms_force[int(ki)] = False

    sorted_sift_frames = sorted(list(transforms_np.keys()))
    logger.info(f"SIFT frames: {sorted_sift_frames}")

    # Map each pose frame to a transform (interpolated/extrapolated)
    frame_to_transform = {}
    for pose_frame in pose_frame_keys:
        if pose_frame in transforms_np:
            frame_to_transform[pose_frame] = transforms_np[pose_frame]
            logger.debug(f"Pose frame {pose_frame} -> direct SIFT frame {pose_frame}")
            continue

        # find nearest before and after
        before = None
        after = None
        for f in sorted_sift_frames:
            if f <= pose_frame:
                before = f
            elif f > pose_frame and after is None:
                after = f
                break

        if before is None and after is None:
            logger.warning(f"No available SIFT transforms to map pose frame {pose_frame}")
            continue
        if before is None:
            frame_to_transform[pose_frame] = transforms_np[after]
            logger.debug(f"Pose frame {pose_frame}: using first SIFT frame {after}")
            continue
        if after is None:
            frame_to_transform[pose_frame] = transforms_np[before]
            logger.debug(f"Pose frame {pose_frame}: using last SIFT frame {before}")
            continue

        # interpolate
        if before == after:
            frame_to_transform[pose_frame] = transforms_np[before]
        else:
            span = after - before
            alpha = float(pose_frame - before) / float(span) if span != 0 else 0.0
            frame_to_transform[pose_frame] = _interpolate_affine(transforms_np[before], transforms_np[after], alpha)
            logger.debug(f"Pose frame {pose_frame}: interpolated between {before} and {after} (alpha={alpha:.3f})")

    # Apply per-frame transforms to pose landmarks
    transformed_poses = {}
    for frame_num in pose_frame_keys:
        if frame_num not in frame_to_transform:
            logger.debug(f"No transform mapped for pose frame {frame_num}, skipping")
            continue
        T = frame_to_transform[frame_num]
        force_flag = transforms_force.get(frame_num, False)
        try:
            pose_data = pose_landmarks[str(frame_num)] if str(frame_num) in pose_landmarks else pose_landmarks.get(frame_num)
            transformed_poses[frame_num] = apply_transform(T, pose_data, force_apply=force_flag)
        except Exception as e:
            logger.exception(f"Error applying transform to pose frame {frame_num}: {e}")

    logger.info(f"Successfully mapped {len(transformed_poses)} pose frames to transformations")

    # Generate video frames with interpolation between pose frames
    total_frames_written = 0
    for i in range(len(pose_frame_keys) - 1):
        f1, f2 = pose_frame_keys[i], pose_frame_keys[i + 1]
        if f1 not in transformed_poses or f2 not in transformed_poses:
            logger.warning(f"Missing transformed pose for frames {f1} or {f2}")
            continue

        pose1, pose2 = transformed_poses[f1], transformed_poses[f2]
        frame_gap = f2 - f1

        # Write first frame
        np.copyto(frame_buffer, ref_img)
        draw_pose(frame_buffer, pose1, line_color, point_color)
        writer.write(frame_buffer)
        total_frames_written += 1

        if frame_gap > 1:
            alphas = np.linspace(1/frame_gap, (frame_gap-1)/frame_gap, frame_gap-1)
            for alpha in alphas:
                np.copyto(frame_buffer, ref_img)
                interp_pose = linear_interpolate_pose(pose1, pose2, alpha)
                draw_pose(frame_buffer, interp_pose, line_color, point_color)
                writer.write(frame_buffer)
                total_frames_written += 1

    # Write final frame if available
    last_key = pose_frame_keys[-1] if pose_frame_keys else None
    if last_key is not None and last_key in transformed_poses:
        np.copyto(frame_buffer, ref_img)
        draw_pose(frame_buffer, transformed_poses[last_key], line_color, point_color)
        writer.write(frame_buffer)
        total_frames_written += 1

    logger.info(f"Generated {total_frames_written} total frames for multi-frame video")
    del frame_buffer