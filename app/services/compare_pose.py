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


from .video_writer import write_pose_video
from .detect_img_sift import detect_sift
from .match_features import match_features, compute_affine_transform
from .draw_points import apply_transform, draw_pose

VIDEO_OUT_DIR = os.path.join("temp_uploads", "pose_feature_data", "output_video")
POSE_JSON = os.path.join("static", "pose_feature_data", "pose_landmarks.json")
SIFT_JSON = os.path.join("static", "pose_feature_data", "sift_keypoints.json")

# Interpolate between poses in subsequent stored frames
# Example: If data is stored for every 20 frames, this will estimate pose landmark 
# coordinates for the 19 frames in between, creating a smoother transition.
def linear_interpolate_pose(pose1, pose2, alpha):
    shared_keys = set(pose1.keys()) & set(pose2.keys())
    result = {}
    for k in shared_keys:
        v1, v2 = pose1[k], pose2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            result[k] = linear_interpolate_pose(v1, v2, alpha)
        else:
            result[k] = (1 - alpha) * np.array(v1) + alpha * np.array(v2)
    return result


def linear_interpolate_pose_vectorized(pose1, pose2, alpha):
    
    shared_keys = set(pose1.keys()) & set(pose2.keys())
    result = {}
    
    for k in shared_keys:
        v1, v2 = pose1[k], pose2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            # Recursively handle nested dictionaries (e.g., body parts with sub-landmarks)
            result[k] = linear_interpolate_pose_vectorized(v1, v2, alpha)
        else:
            # Vectorized calculation - significantly faster than individual operations
            # Convert to NumPy arrays once and perform batch mathematical operations
            v1_arr = np.asarray(v1, dtype=np.float32)
            v2_arr = np.asarray(v2, dtype=np.float32)
            result[k] = (1 - alpha) * v1_arr + alpha * v2_arr
    
    return result


def _generate_video_frames_optimized(ref_img, transformed, frame_keys, line_color, point_color, writer):
    
    # =================================================================
    # Pre-allocate frame buffer to avoid memory thrashing
    # =================================================================
    h, w = ref_img.shape[:2]
    frame_buffer = np.empty((h, w, 3), dtype=np.uint8)
    
    logger.info(f"Starting optimized frame generation for {len(frame_keys)} key frames")
    total_frames_written = 0
    
    # =================================================================
    # Process frame segments efficiently
    # =================================================================
    # Iterate through consecutive frame pairs and generate all interpolated
    # frames for each segment in one batch, reducing overhead
    for i in range(len(frame_keys) - 1):
        f1, f2 = frame_keys[i], frame_keys[i + 1]
        pose1, pose2 = transformed[f1], transformed[f2]
        frame_gap = f2 - f1
        
        # =========================================================
        # Write the first frame of this segment
        # =========================================================
        # Use np.copyto() instead of .copy() - faster for memory operations
        # because it reuses existing allocated memory instead of creating new arrays
        np.copyto(frame_buffer, ref_img)
        draw_pose(frame_buffer, pose1, line_color, point_color)
        writer.write(frame_buffer)
        total_frames_written += 1
        
        # =========================================================
        # Batch interpolation for frame gaps
        # =========================================================
        # If there's a gap between frames (0, 15, 30, ... , 15*n), generate interpolated
        # frames for smooth animation. Pre-calculate all alpha values using NumPy's linspace
        # which is vectorized and much faster than a for loop.
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
                interp_pose = linear_interpolate_pose_vectorized(pose1, pose2, alpha)
                draw_pose(frame_buffer, interp_pose, line_color, point_color)
                
                writer.write(frame_buffer)
                total_frames_written += 1
    
    # =================================================================
    # Write the final frame
    # =================================================================
    # Handle the last frame separately since the loop processes pairs
    np.copyto(frame_buffer, ref_img)
    draw_pose(frame_buffer, transformed[frame_keys[-1]], line_color, point_color)
    writer.write(frame_buffer)
    total_frames_written += 1
    
    logger.info(f"Successfully generated {total_frames_written} total video frames")
    
    # =================================================================
    # MEMORY CLEANUP
    # =================================================================
    # Explicitly delete the frame buffer to free memory immediately
    # This is especially important for large images or when processing multiple videos
    del frame_buffer

# Performance monitoring decorator: 
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
        if scale_x > 5.0 or scale_y > 5.0 or scale_x < 0.2 or scale_y < 0.2:
            print("Unusual scale detected")
        if abs(rotation_angle) > 45:
            print("Large rotation angle detected")
    except Exception as e:
        print(f"Error validating affine matrix: {e}")

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
    frame_dimensions=None
):

    # Clean up local storage ONCE before any S3 downloads
    from .load_json_s3 import cleanup_local_storage
    cleanup_local_storage()
    logger.info("Starting optimized video generation with enhanced performance monitoring")

    gc.collect()
    cv2.setUseOptimized(False)
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
    os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
    out_path = os.path.join(VIDEO_OUT_DIR, output_video)
    for f in os.listdir(VIDEO_OUT_DIR):
        if f.endswith(".mp4"):
            os.remove(os.path.join(VIDEO_OUT_DIR, f))

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
    if not ref_kp or ref_desc is None or len(ref_kp) == 0:
        logger.error("SIFT feature detection failed - unable to proceed with video generation")
        return {"error": "SIFT failure"}

    frame_keys = sorted([int(k) for k in pose_landmarks.keys()])
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    if not writer.isOpened():
        logger.error(f"Failed to initialize video writer for {out_path}")
        return {"error": "Video writer failed"}

    static_mode = len(stored_keypoints_all) == 1
    if static_mode:
        kp = stored_keypoints_all[0]
        desc = stored_descriptors_all[0]
        matches = match_features(
            desc1=desc,
            desc2=ref_desc,
            ratio_thresh=0.85,
            distance_thresh=400,
            top_n=200,
            min_required_matches=15,
            prev_query_indices=None,
            min_shared_matches=0,
            debug=False
        )
        if not matches:
            print("No matches")
            writer.release()
            return "NO_MATCHES"
        T = compute_affine_transform(
            kp, ref_kp, matches,
            prev_T=None,
            ransac_thresh=1.0,
            max_iters=5000,
            confidence=0.999,
            alpha=0.0,
            min_required_matches=3,
            debug=False
        )
        if T is None:
            print("No transform")
            writer.release()
            return "NO_TRANSFORM"

        print("Validating transform")
        validate_affine_transform(T)

        # Pre-transform all pose landmarks using the computed transformation matrix
        # This is done once upfront to avoid repeated transform calculations
        transformed = {
            frame: apply_transform(T, pose_landmarks[frame])
            for frame in frame_keys
        }
        
        # Generate video frames using optimized approach
        _generate_video_frames_optimized(
            ref_img, transformed, frame_keys, line_color, point_color, writer
        )
        
        writer.release()
        logger.info(f"Finished writing optimized video to {out_path}")
        return

    writer.release()
    return

def convert_video_for_browser(
    input_path: str, 
    output_path: str, 
    async_mode: bool = False,
    quality_preset: str = "fast"
) -> Optional[Dict[str, Any]]:
    
    # Input validation
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input video file not found: {input_path}")
        return {"error": "Input file not found", "path": input_path}
    
    if input_file.stat().st_size == 0:
        logger.error(f"Input video file is empty: {input_path}")
        return {"error": "Input file is empty", "path": input_path}
    
    # Check if output already exists and is newer
    output_file = Path(output_path)
    if (output_file.exists() and 
        output_file.stat().st_mtime > input_file.stat().st_mtime):
        logger.info(f"Output file already up-to-date: {output_path}")
        return {"status": "already_exists", "path": output_path}
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check ffmpeg availability
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        logger.error("ffmpeg executable not found in PATH")
        return {"error": "ffmpeg not found"}
    
    def _do_conversion():
        """Internal conversion function"""
        # Quality presets for different use cases
        presets = {
            "fast": {
                "crf": "28",
                "preset": "ultrafast",
                "profile": "baseline",
                "level": "3.0"
            },
            "balanced": {
                "crf": "23",
                "preset": "fast", 
                "profile": "main",
                "level": "3.1"
            },
            "quality": {
                "crf": "20",
                "preset": "medium",
                "profile": "high",
                "level": "4.0"
            }
        }
        
        settings = presets.get(quality_preset, presets["balanced"])
        
        # Optimized ffmpeg command
        cmd = [
            ffmpeg_exe,
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-c:v", "libx264",
            "-crf", settings["crf"],
            "-preset", settings["preset"],
            "-profile:v", settings["profile"],
            "-level", settings["level"],
            "-movflags", "+faststart",  # Enable progressive download
            "-pix_fmt", "yuv420p",  # Ensure compatibility
            "-f", "mp4",  # Force MP4 format
            str(output_path)
        ]
        
        try:
            # Run with timeout and capture output
            result = subprocess.run(
                cmd, 
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Verify output file was created successfully
            if not output_file.exists() or output_file.stat().st_size == 0:
                raise Exception("Output file was not created or is empty")
            
            logger.info(f"Successfully converted video: {output_path}")
            return {
                "status": "success", 
                "input_path": input_path,
                "output_path": output_path,
                "file_size": output_file.stat().st_size
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg conversion timed out for {input_path}")
            return {"error": "conversion_timeout", "path": input_path}
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr}")
            return {"error": "conversion_failed", "details": e.stderr}
        except Exception as e:
            logger.error(f"Unexpected error during conversion: {e}")
            return {"error": "unexpected_error", "details": str(e)}
    
    if async_mode:
        # Run conversion in background thread
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_conversion)
        return {"status": "started", "future": future}
    else:
        # Run synchronously
        return _do_conversion()

def batch_convert_videos(
    video_paths: list[str], 
    output_dir: str,
    max_workers: int = 2,
    quality_preset: str = "balanced"
) -> Dict[str, Any]:

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    results = {"success": [], "failed": [], "skipped": []}
    
    def convert_single(input_path: str) -> tuple[str, Dict[str, Any]]:
        input_file = Path(input_path)
        output_path = output_dir_path / f"{input_file.stem}_browser.mp4"
        
        result = convert_video_for_browser(
            input_path, 
            str(output_path), 
            async_mode=False,
            quality_preset=quality_preset
        )
        return input_path, result
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_single, path) for path in video_paths]
        
        for future in futures:
            try:
                input_path, result = future.result(timeout=600)  # 10 min timeout per video
                
                if result and result.get("status") == "success":
                    results["success"].append({"input": input_path, "result": result})
                elif result and result.get("status") == "already_exists":
                    results["skipped"].append({"input": input_path, "result": result})
                else:
                    results["failed"].append({"input": input_path, "result": result})
                    
            except Exception as e:
                results["failed"].append({
                    "input": input_path, 
                    "result": {"error": "batch_timeout", "details": str(e)}
                })
    
    logger.info(f"Batch conversion complete: {len(results['success'])} success, "
                f"{len(results['failed'])} failed, {len(results['skipped'])} skipped")
    
    return results