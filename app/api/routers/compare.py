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
import uuid
import base64

from app.storage.s3.load_json_s3 import load_pose_data_from_path, load_sift_data_from_path
from app.pipelines.video_tripod import generate_video
from app.pipelines.video_moving import generate_video_multiframe
from app.jobs.job_manager import submit_job, get_job
from app.transform.draw_points import rgb_to_bgr
from app.video.convert import convert_video_for_browser

VIDEO_OUT_DIR = os.path.join("temp_uploads", "pose_feature_data", "output_video")

# Set up logger for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()

# Receives image and parameters from the client that are used to compare 
# the image against stored climbing data to generate a video.
@router.post("/compare-image")
async def compare_image(
    image: UploadFile = File(None),
    built_in_image: str = Form(""),
    s3_folders: List[str] = Form(...),
    sift_left: float = Form(20.0),
    sift_right: float = Form(20.0),
    sift_up: float = Form(20.0),
    sift_down: float = Form(20.0),
    line_color: str = Form("100,255,0"),
    point_color: str = Form("0,100,255"),
    fps: int = Form(24),
): 
    print("[DIAGNOSTIC] REAL compare-image endpoint called")
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
            
        # In app/routers/compare.py, modify the data loading section:

        all_pose_data = {}
        all_sift_data = [] 
        frame_dimensions = None
        is_multi_frame_data = False

        # Load pose and SIFT data from S3 folders
        for folder in s3_folders:
            key = folder.replace("s3://route-keypoints/", "").strip("/")
            try:
                pose = load_pose_data_from_path(key)
                sift_data, sift_frame_dims, is_multi_frame = load_sift_data_from_path(key)
                
                # Use frame dimensions from the first SIFT file that has them
                if frame_dimensions is None and sift_frame_dims:
                    frame_dimensions = sift_frame_dims
                    logger.info(f"Using frame dimensions from {key}: {frame_dimensions}")
                
                # Track if any data is multi-frame
                if is_multi_frame:
                    is_multi_frame_data = True
                    logger.info(f"Multi-frame SIFT data detected in {key}")
                
            except Exception as e:
                logger.exception(f"Error loading pose/SIFT data from S3 for {key}")
                continue
            
            # Combine pose data
            for frame, items in pose.items():
                all_pose_data.setdefault(frame, []).extend(items)
            
            # Store SIFT data (format depends on whether it's multi-frame)
            all_sift_data.append({
                "data": sift_data,
                "is_multi_frame": is_multi_frame,
                "source": key
            })

        # Convert pose data keys to integers
        try:
            all_pose_data = {int(k): v for k, v in all_pose_data.items()}
        except Exception as e:
            logger.exception("Error converting pose data keys to int")

        # Parse color parameters before using them
        def parse_color(s):
            try:
                return tuple(int(x) for x in s.split(","))
            except Exception as e:
                print(f"Error parsing color string '{s}': {e}")
                return (100, 255, 0)
        
        line_color_tuple = rgb_to_bgr(parse_color(line_color))
        point_color_tuple = rgb_to_bgr(parse_color(point_color))

        # Clean up old video files BEFORE generation (not after!)
        abs_video_out_dir = os.path.abspath(VIDEO_OUT_DIR)
        raw_path = os.path.join(abs_video_out_dir, "output_video.mp4")
        browser_ready = os.path.join(abs_video_out_dir, "output_video_browser.mp4")
        
        # Clean up old files
        for out_file in [raw_path, browser_ready]:
            if os.path.exists(out_file):
                try:
                    os.remove(out_file)
                    logger.info(f"Removed old video file before generation: {out_file}")
                except Exception as e:
                    logger.exception(f"Failed to delete old output video: {out_file}")

        # Clear any potential SIFT caches to prevent stale data between requests
        gc.collect()

        # Instead of running synchronously, submit a background job.
        job_input = {
            "temp_image_path": temp_image_path,
            "all_pose_data": all_pose_data,
            "all_sift_data": all_sift_data,
            "frame_dimensions": frame_dimensions,
            "is_multi_frame_data": is_multi_frame_data,
            "sift_left": sift_left,
            "sift_right": sift_right,
            "sift_up": sift_up,
            "sift_down": sift_down,
            "line_color_tuple": line_color_tuple,
            "point_color_tuple": point_color_tuple,
            "fps": fps,
        }

        def _job_runner(job_id, opts):
            try:

                # Generate main video
                if opts.get("is_multi_frame_data"):

                    video_result = generate_video_multiframe(
                        image_path=opts["temp_image_path"],
                        pose_landmarks=opts["all_pose_data"],
                        sift_data_all=opts["all_sift_data"],
                        sift_left=opts["sift_left"],
                        sift_right=opts["sift_right"],
                        sift_up=opts["sift_up"],
                        sift_down=opts["sift_down"],
                        line_color=opts["line_color_tuple"],
                        point_color=opts["point_color_tuple"],
                        frame_dimensions=opts.get("frame_dimensions"),
                        fps=opts.get("fps", 24),
                        job_id=job_id
                    )
                else:
                    all_sift_keypoints = []
                    all_sift_descriptors = []
                    for sift_entry in opts.get("all_sift_data", []):
                        if not sift_entry.get("is_multi_frame"):
                            kps_all, descs_all = sift_entry["data"]
                            all_sift_keypoints.extend(kps_all)
                            all_sift_descriptors.extend(descs_all)

                    video_result = generate_video(
                        image_path=opts["temp_image_path"],
                        pose_landmarks=opts["all_pose_data"],
                        stored_keypoints_all=all_sift_keypoints,
                        stored_descriptors_all=all_sift_descriptors,
                        sift_left=opts["sift_left"],
                        sift_right=opts["sift_right"],
                        sift_up=opts["sift_up"],
                        sift_down=opts["sift_down"],
                        line_color=opts["line_color_tuple"],
                        point_color=opts["point_color_tuple"],
                        frame_dimensions=opts.get("frame_dimensions"),
                        fps=opts.get("fps", 24),
                        job_id=job_id
                    )

                # Convert to browser-ready format
                abs_video_out_dir = os.path.abspath(VIDEO_OUT_DIR)
                raw_path = os.path.join(abs_video_out_dir, "output_video.mp4")
                browser_ready = os.path.join(abs_video_out_dir, "output_video_browser.mp4")

                convert_result = convert_video_for_browser(
                    input_path=raw_path,
                    output_path=browser_ready,
                    async_mode=False,
                    quality_preset="fast"
                )

                print(f"Browser conversion result: {convert_result}")
                return video_result
            finally:
                try:
                    if os.path.exists(opts.get("temp_image_path", "")):
                        os.remove(opts.get("temp_image_path"))
                except Exception:
                    pass

        logger.info(f"Submitting compare job with input: {job_input}")
        job_id = submit_job(_job_runner, job_input)
        status_url = f"/api/compare-status/{job_id}"
        logger.info(f"Enqueued compare job {job_id}, status at {status_url}")
        return JSONResponse(status_code=202, content={"job_id": job_id, "status_url": status_url})


    except HTTPException as he:
        print("HTTPException in compare_image route:", he)
        raise he
    except Exception as e:
        print("Error in compare_image route:", e)
        logger.exception("Error in compare_image route")
        return JSONResponse(status_code=500, content={"error": "Failed to process image."})


@router.get("/compare-status/{job_id}")
def compare_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    resp = {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "error": job.get("error"),
        "result": job.get("result"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
    }

    # Always include video_url if job is successful and video exists
    if job.get("status") == "success":
        abs_video_out_dir = os.path.abspath(VIDEO_OUT_DIR)
        browser_ready = os.path.join(abs_video_out_dir, "output_video_browser.mp4")
        if os.path.exists(browser_ready) and os.path.getsize(browser_ready) > 0:
            resp["video_url"] = "/static/pose_feature_data/output_video/output_video_browser.mp4"
        else:
            resp["video_url"] = None
    else:
        resp["video_url"] = None

    return JSONResponse(resp)

@router.get("/current-image")
def current_image():
    try:
        with open("temp_uploads/pose_feature_data/current.jpg", "rb") as f:
            img_data = f.read()
        return JSONResponse(content={"image": base64.b64encode(img_data).decode('utf-8')})
    except Exception as e:
        logger.error(f"Error loading current image: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to load current image."})