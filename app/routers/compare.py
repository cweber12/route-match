# app/routers/compare.py
# -----------------------------------------------------------------------------------------
# This file handles the comparison of climbing route images using SIFT and pose estimation.
# It allows users to upload images and compare them against stored climbing data (pose/SIFT)
# for a route.
# -----------------------------------------------------------------------------------------

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
import cv2
import numpy as np
import logging
import gc

from app.services.load_json_s3 import load_pose_data_from_path, load_sift_data_from_path
from app.services.compare_pose import (
    create_video_from_static_image_streamed,  
    convert_video_for_browser,
    VIDEO_OUT_DIR,
)
from app.services.draw_points import rgb_to_bgr

# Set up logger for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()

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
            # If the built-in image is an S3 path, download it
            if built_in_image.startswith("s3://"):
                import boto3
                from urllib.parse import urlparse
                s3 = boto3.client("s3")
                parsed = urlparse(built_in_image)
                bucket = parsed.netloc
                key = parsed.path.lstrip("/")
                try:
                    # download the s3 image to a temp file
                    with open(temp_image_path, "wb") as f:
                        s3.download_fileobj(bucket, key, f)
                        f.flush()  # Ensure Python sends data to OS
                        os.fsync(f.fileno())  # Ensure OS writes data to disk
                    print(f"Downloaded S3 image: {built_in_image} -> {temp_image_path}")
                except Exception as e:
                    raise HTTPException(404, f"Failed to download S3 image: {built_in_image}")
                if not os.path.exists(temp_image_path):
                    raise HTTPException(500, "S3 image file not found after download.")
                file_size = os.path.getsize(temp_image_path)
                if file_size == 0:
                    raise HTTPException(500, "S3 image file is empty after download.")
                
            else:
                # Or if the built-in-image is a local static file, copy it
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
            gc.collect()

        # Ensure the temp image file is properly written and flushed
        if os.path.exists(temp_image_path):
            # Force filesystem sync to ensure file is fully written
            import time
            time.sleep(0.1)  # Small delay to ensure file system consistency
            
        all_pose_data = {}
        all_sift_keypoints = []
        all_sift_descriptors = []
        
        # Load pose and SIFT data from S3 folders
        for folder in s3_folders:
            key = folder.replace("s3://route-keypoints/", "").strip("/")
            try:
                pose = load_pose_data_from_path(key)
                sift_kps, sift_descs, frame_dimensions = load_sift_data_from_path(key)
            except Exception as e:
                logger.exception(f"Error loading pose/SIFT data from S3 for {key}")
                continue
            for frame, items in pose.items():
                all_pose_data.setdefault(frame, []).extend(items)
            all_sift_keypoints.extend(sift_kps)
            all_sift_descriptors.extend(sift_descs)
        try:
            all_pose_data = {int(k): v for k, v in all_pose_data.items()}
        except Exception as e:
            logger.exception("Error converting pose data keys to int")

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
                    logger.exception(f"Failed to delete old output video: {out_file}")

        # Clear any potential SIFT caches to prevent stale data between requests
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
                point_color=point_color_tuple,
                frame_dimensions=frame_dimensions,
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

