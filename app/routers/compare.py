from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
import cv2
import numpy as np
import logging

from app.services.load_json_s3 import load_pose_data_from_path, load_sift_data_from_path
from app.services.compare_pose import (
    create_video_from_static_image_streamed,  
    convert_video_for_browser,
    VIDEO_OUT_DIR,
)
from app.services.draw_points import rgb_to_bgr

# Set up logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Health check for filesystem access
@router.get("/health-check-fs")
async def health_check_fs():
    import stat
    import gc
    results = {}
    temp_dir = "temp_uploads"
    output_dir = os.path.join("temp_uploads", "pose_feature_data", "output_video")
    for dir_path in [temp_dir, output_dir]:
        results[dir_path] = {}
        try:
            # Check existence and permissions
            exists = os.path.exists(dir_path)
            perms = None
            if exists:
                perms = stat.filemode(os.stat(dir_path).st_mode)
            results[dir_path]["exists"] = exists
            results[dir_path]["permissions"] = perms
            # Try to create, write, read, and delete a temp file
            test_file = os.path.join(dir_path, "health_check_test.txt")
            try:
                with open(test_file, "w") as f:
                    f.write("health check")
                results[dir_path]["write"] = True
            except Exception as e:
                results[dir_path]["write"] = f"ERROR: {e}"
            try:
                with open(test_file, "r") as f:
                    content = f.read()
                results[dir_path]["read"] = content
            except Exception as e:
                results[dir_path]["read"] = f"ERROR: {e}"
            try:
                os.remove(test_file)
                results[dir_path]["delete"] = True
            except Exception as e:
                results[dir_path]["delete"] = f"ERROR: {e}"
        except Exception as e:
            results[dir_path]["error"] = str(e)
    
    try: # Try to force garbage collection
        gc.collect()
        results["gc"] = "collected"
    except Exception as e:
        results["gc"] = f"ERROR: {e}"
    # Log results
    print("HEALTH CHECK FS RESULTS:", results)
    return JSONResponse(results)

router = APIRouter()

# Ensure all keypoints are valid (x, y) tuples, skip invalid ones
def sanitize_skeleton(skeleton):
    sanitized = {}
    for k, pt in skeleton.items():
        if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
            try:
                # Ensure pt is iterable and has at least 2 elements
                x, y = pt[0], pt[1]
                if isinstance(x, (int, float, np.integer, np.floating)) and isinstance(y, (int, float, np.integer, np.floating)):
                    sanitized[k] = (int(x), int(y))
            except (TypeError, ValueError):
                pass  # Skip invalid keypoints
    
    # Return verified coordinates
    return sanitized

# Receives an image w/ defined SIFT bbox from the client
@router.post("/compare-image")
async def compare_image(
    image: UploadFile = File(None),
    built_in_image: str = Form(""),
    s3_folders: List[str] = Form(...),
    sift_left: float = Form(20.0),
    sift_right: float = Form(20.0),
    sift_up: float = Form(20.0),
    sift_down: float = Form(20.0),
    pose_lm_in: str = Form(""),
    sift_kp_in: str = Form(""),
    line_color: str = Form("100,255,0"),
    point_color: str = Form("0,100,255"),
):
    import uuid
    temp_image_path = None
    try:
        # Save uploaded image to a unique temp file if provided
        if image is not None:
            ext = os.path.splitext(image.filename)[1] if image.filename else ".jpg"
            temp_image_path = os.path.join("temp_uploads", f"compare_image_{uuid.uuid4().hex}{ext}")
            with open(temp_image_path, "wb") as buffer:
                image.file.seek(0)
                shutil.copyfileobj(image.file, buffer)
                buffer.flush()
                os.fsync(buffer.fileno())
            print(f"Saved uploaded image: {temp_image_path}")
        elif built_in_image:
            # Use built-in image path logic here if needed
            temp_image_path = built_in_image  # Or however you resolve built-in images
        else:
            raise HTTPException(400, "No image provided.")

        # Save image to temp file, always flush and close
        if built_in_image:
            if built_in_image.startswith("s3://"):
                import boto3
                from urllib.parse import urlparse
                s3 = boto3.client("s3")
                parsed = urlparse(built_in_image)
                bucket = parsed.netloc
                key = parsed.path.lstrip("/")
                try:
                    with open(temp_image_path, "wb") as f:
                        s3.download_fileobj(bucket, key, f)
                        f.flush()
                        os.fsync(f.fileno())
                    print(f"Downloaded S3 image: {built_in_image} -> {temp_image_path}")
                except Exception as e:
                    raise HTTPException(404, f"Failed to download S3 image: {built_in_image}")
                if not os.path.exists(temp_image_path):
                    raise HTTPException(500, "S3 image file not found after download.")
                file_size = os.path.getsize(temp_image_path)
                if file_size == 0:
                    raise HTTPException(500, "S3 image file is empty after download.")
                
            else:
                static_image_path = os.path.join("static", "images", built_in_image)
                if not os.path.exists(static_image_path):
                    raise HTTPException(404, f"Built-in image not found: {built_in_image}")
                shutil.copyfile(static_image_path, temp_image_path)
                print(f"Copied built-in image: {static_image_path} -> {temp_image_path}")
                file_size = os.path.getsize(temp_image_path)

                if file_size == 0:
                    raise HTTPException(500, "Static image file is empty.")
        elif image is not None:
            # Already saved uploaded image above, just check file size
            file_size = os.path.getsize(temp_image_path)
            if file_size == 0:
                raise HTTPException(500, "Uploaded image file is empty.")
        else:
            raise HTTPException(400, "No image provided.")

        # Validate image load
        image_cv2 = None
        try:
            image_cv2 = cv2.imread(temp_image_path)
            if image_cv2 is None:
                raise HTTPException(500, "Failed to read uploaded image.")
        except Exception as e:
            logger.exception("Error loading image with cv2.imread")
            raise HTTPException(500, f"Error loading image_cv2: {e}")
        finally:
            # Release image_cv2 and force garbage collection
            del image_cv2
            import gc
            gc.collect()

        # Ensure the temp image file is properly written and flushed
        if os.path.exists(temp_image_path):
            # Force filesystem sync to ensure file is fully written
            import time
            time.sleep(0.1)  # Small delay to ensure file system consistency
            
        all_pose_data = {}
        all_sift_keypoints = []
        all_sift_descriptors = []

        for folder in s3_folders:
            key = folder.replace("s3://route-keypoints/", "").strip("/")
            try:
                pose = load_pose_data_from_path(key)
                sift_kps, sift_descs = load_sift_data_from_path(key)
            except Exception as e:
                logger.exception(f"Error loading pose/SIFT data from S3 for {key}")
                print(f"Error loading pose/SIFT data from S3 for {key}: {e}")
                continue
            for frame, items in pose.items():
                all_pose_data.setdefault(frame, []).extend(items)
            all_sift_keypoints.extend(sift_kps)
            all_sift_descriptors.extend(sift_descs)
        try:
            all_pose_data = {int(k): v for k, v in all_pose_data.items()}
        except Exception as e:
            logger.exception("Error converting pose data keys to int")
            print(f"Error converting pose data keys to int: {e}")

        def parse_color(s):
            try:
                return tuple(int(x) for x in s.split(","))
            except Exception as e:
                print(f"Error parsing color string '{s}': {e}")
                return (100, 255, 0)
        line_color_tuple = rgb_to_bgr(parse_color(line_color))
        point_color_tuple = rgb_to_bgr(parse_color(point_color))

        # Remove output video files before generating new ones
        raw_path = os.path.join(VIDEO_OUT_DIR, "output_video.mp4")
        browser_ready = os.path.join(VIDEO_OUT_DIR, "output_video_browser.mp4")
        for out_file in [raw_path, browser_ready]:
            if os.path.exists(out_file):
                try:
                    os.remove(out_file)
                except Exception as e:
                    print(f"Failed to delete old output video: {out_file}", e)

        # Clear any potential SIFT caches to prevent stale data between requests
        import gc
        gc.collect()
        
        try:
            result = create_video_from_static_image_streamed(
                image_path=temp_image_path,
                pose_landmarks=all_pose_data,
                stored_keypoints_all=all_sift_keypoints,
                stored_descriptors_all=all_sift_descriptors,
                sift_left=sift_left,
                sift_right=sift_right,
                sift_up=sift_up,
                sift_down=sift_down,
                line_color=line_color_tuple,
                point_color=point_color_tuple
            )
            if result == "NO_MATCHES":
                raise HTTPException(422, "No matching features found between the uploaded image and the climbing route. Please try a different image or route.")
            elif result == "NO_TRANSFORM":
                raise HTTPException(422, "Unable to compute transformation between images. Please try a different image or route.")
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.exception("Error in video generation")
            raise HTTPException(500, f"Error in video generation: {e}")

        raw_path = os.path.join(VIDEO_OUT_DIR, "output_video.mp4")
        browser_ready = os.path.join(VIDEO_OUT_DIR, "output_video_browser.mp4")
        # Check that output_video.mp4 exists and is non-empty before conversion
        try:
            if not os.path.exists(raw_path):
                raise HTTPException(500, f"Video file not created: {raw_path}")
            file_size = os.path.getsize(raw_path)
            if file_size == 0:
                raise HTTPException(500, f"Output video file is empty: {raw_path}")
        except Exception as e:
            logger.exception("Error validating output video")
            raise HTTPException(500, f"Error validating output video: {e}")

        # Validation before FFmpeg conversion
        try:
            if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
                raise HTTPException(500, "Output video not created or is empty.")
            convert_video_for_browser(raw_path, browser_ready)
        except Exception as e:
            logger.exception("Error converting video for browser")
            raise HTTPException(500, f"Error converting video for browser: {e}")

        try:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        except Exception as e:
            raise HTTPException(500, f"Failed to delete temp image at end of request: {temp_image_path}: {e}")
        return JSONResponse({
            "message": "Video created successfully.",
            "video_url": "/static/pose_feature_data/output_video/output_video_browser.mp4"
        })

    except HTTPException as he:
        print("HTTPException in compare_image route:", he)
        raise he
    except Exception as e:
        print("Error in compare_image route:", e)
        logger.exception("Error in compare_image route")
        return JSONResponse(status_code=500, content={"error": "Failed to process image."})

def transform_poses_to_image(
    image_path,
    pose_landmarks,
    stored_keypoints_all,
    stored_descriptors_all,
    sift_left=20.0,
    sift_right=20.0,
    sift_up=20.0,
    sift_down=20.0
):

    ref_img = cv2.imread(image_path)
    if ref_img is None:
        return {}

    sift_config = {
        "nfeatures": 2000,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6
    }

    from app.services.detect_img_sift import detect_sift
    from app.services.match_features import match_features, compute_affine_transform
    from app.services.draw_points import apply_transform

    h_full, w_full = ref_img.shape[:2]
    x1_s = int(sift_left / 100 * w_full)
    y1_s = int(sift_up / 100 * h_full)
    x2_s = int(w_full - (sift_right / 100) * w_full)
    y2_s = int(h_full - (sift_down / 100) * h_full)

    bbox = (x1_s, y1_s, x2_s, y2_s)
    print(f"Using SIFT bbox: {bbox}")

    ref_kp, ref_desc = detect_sift(ref_img, sift_config=sift_config, use_clahe=False, bbox=bbox)

    transformed = {}
    prev_T = None
    prev_query_indices = set()

    frame_keys = sorted(pose_landmarks.keys())
    for i, frame_num in enumerate(frame_keys):
        if len(stored_keypoints_all) == 1:
            kp = stored_keypoints_all[0]
            desc = stored_descriptors_all[0]
        elif i < len(stored_keypoints_all):
            kp = stored_keypoints_all[i]
            desc = stored_descriptors_all[i]
        else:
            continue

        print(f"Processing frame {frame_num}")

        matches = match_features(
            desc, ref_desc,
            prev_query_indices=prev_query_indices,
            min_shared_matches=0,
            ratio_thresh=0.75,
            distance_thresh=300,
            min_required_matches=10,
            top_n=150,
            debug=False
        )

        if not matches:
            print(f"No matches found for frame {frame_num}")
            continue

        shared = [m for m in matches if m.queryIdx in prev_query_indices] if prev_query_indices else matches
        use_matches = shared if len(shared) >= 5 else matches

        T = compute_affine_transform(
            kp, ref_kp, use_matches,
            prev_T=prev_T,
            alpha=0.9,
            debug=False
        )

        if T is None:
            print(f"Failed to compute transformation matrix for frame {frame_num}")
            continue

        transformed[frame_num] = apply_transform(T, pose_landmarks[frame_num])
        prev_T = T
        prev_query_indices = set(m.queryIdx for m in use_matches)

    return transformed
