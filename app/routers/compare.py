# app/api/compare_image.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
import cv2
import numpy as np

from app.services.compare_pose import (
    load_pose_data_from_path,
    load_sift_data_from_path,
    create_video_from_static_image,  
    convert_video_for_browser,
    VIDEO_OUT_DIR,
    linear_interpolate_pose
)
from app.services.draw_points import draw_pose
from app.utils.json_loader import load_json_from_path

router = APIRouter()

def sanitize_skeleton(skeleton):
    """Ensure all keypoints are valid (x, y) tuples, skip invalid ones"""
    sanitized = {}
    for k, pt in skeleton.items():
        if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
            try:
                x, y = pt[0], pt[1]
                if isinstance(x, (int, float, np.integer, np.floating)) and isinstance(y, (int, float, np.integer, np.floating)):
                    sanitized[k] = (int(x), int(y))
            except (TypeError, ValueError):
                pass  # Skip invalid keypoints
        # Skip scalar values entirely - don't create fake coordinates
    return sanitized

@router.post("/compare-image")
async def compare_image(
    image: UploadFile = File(...),
    s3_folders: List[str] = Form(...),
    sift_left: float = Form(20.0),
    sift_right: float = Form(20.0),
    sift_up: float = Form(20.0),
    sift_down: float = Form(20.0),
    pose_lm_in: str = Form(""),
    sift_kp_in: str = Form(""),
):
    try:
        print("Starting compare_image route")
        # Clear temp_uploads folder before saving new image
        temp_dir = "temp_uploads"
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        # Save uploaded image to temp storage
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, "compare_image.jpg")
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        print(f"Saved uploaded image: {temp_image_path}")

        image_cv2 = cv2.imread(temp_image_path)
        print(f"Loaded image_cv2: {type(image_cv2)}, shape: {getattr(image_cv2, 'shape', None)}")
        if image_cv2 is None:
            print("Failed to read uploaded image with cv2.imread")
            raise HTTPException(500, "Failed to read uploaded image.")

        # PREVIEW MODE: use local JSONs if provided
        if pose_lm_in and sift_kp_in:
            print(f"Preview mode: pose_lm_in={pose_lm_in}, sift_kp_in={sift_kp_in}")
            pose_path = os.path.join("temp_uploads", pose_lm_in)
            sift_path = os.path.join("temp_uploads", sift_kp_in)
            print(f"Pose path: {pose_path}, SIFT path: {sift_path}")
            if not os.path.exists(pose_path) or not os.path.exists(sift_path):
                print("Local pose or sift JSON not found.")
                raise HTTPException(400, "Local pose or sift JSON not found.")

            all_pose_data = load_json_from_path(pose_path)
            print(f"Loaded pose data keys: {list(all_pose_data.keys())}")
            sift_data = load_json_from_path(sift_path)
            all_sift_keypoints = sift_data.get("keypoints", [])
            all_sift_descriptors = sift_data.get("descriptors", [])
            print(f"SIFT keypoints: {len(all_sift_keypoints)}, descriptors: {len(all_sift_descriptors)}")

            # Convert keys to int for consistency
            all_pose_data = {int(k): v for k, v in all_pose_data.items()}
            print(f"Pose data keys after int conversion: {list(all_pose_data.keys())}")

            print("Calling create_video_from_static_image (preview mode)")
            create_video_from_static_image(
                image_path=temp_image_path,
                pose_landmarks=all_pose_data,
                stored_keypoints_all=all_sift_keypoints,
                stored_descriptors_all=all_sift_descriptors
            )
            raw_path = os.path.join(VIDEO_OUT_DIR, "output_video.mp4")
            browser_ready = os.path.join(VIDEO_OUT_DIR, "output_video_browser.mp4")
            print(f"Converting video for browser: {raw_path} -> {browser_ready}")
            convert_video_for_browser(raw_path, browser_ready)

            print("Returning preview video response")
            # Delete the uploaded image after video creation
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return JSONResponse({
                "message": "Preview video created successfully.",
                "video_url": "/static/pose_feature_data/output_video/output_video_browser.mp4"
            })

        # PRODUCTION MODE: load from S3 folders
        print("Production mode: loading from S3 folders")
        all_pose_data = {}
        all_sift_keypoints = []
        all_sift_descriptors = []

        for folder in s3_folders:
            key = folder.replace("s3://route-keypoints/", "").strip("/")
            print(f"Processing S3 folder: {key}")

            pose = load_pose_data_from_path(key)
            print(f"Loaded pose data from S3: {len(pose)} frames")
            sift_kps, sift_descs = load_sift_data_from_path(key)
            print(f"Loaded SIFT data from S3: {len(sift_kps)} keypoints, {len(sift_descs)} descriptors")

            for frame, items in pose.items():
                all_pose_data.setdefault(frame, []).extend(items)

            all_sift_keypoints.extend(sift_kps)
            all_sift_descriptors.extend(sift_descs)

        print(f"All pose data keys before int conversion: {list(all_pose_data.keys())}")
        all_pose_data = {int(k): v for k, v in all_pose_data.items()}
        print(f"All pose data keys after int conversion: {list(all_pose_data.keys())}")
        print(f"Total SIFT keypoints: {len(all_sift_keypoints)}, descriptors: {len(all_sift_descriptors)}")

        print("Calling create_video_from_static_image(production mode)")
        create_video_from_static_image(
            image_path=temp_image_path,
            pose_landmarks=all_pose_data,
            stored_keypoints_all=all_sift_keypoints,
            stored_descriptors_all=all_sift_descriptors
        )
        raw_path = os.path.join(VIDEO_OUT_DIR, "output_video.mp4")
        browser_ready = os.path.join(VIDEO_OUT_DIR, "output_video_browser.mp4")
        print(f"Converting video for browser: {raw_path} -> {browser_ready}")
        convert_video_for_browser(raw_path, browser_ready)

        print("Returning production video response")
        # Delete the uploaded image after video creation
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return JSONResponse({
            "message": "Comparison video created successfully.",
            "video_url": "/static/pose_feature_data/output_video/output_video_browser.mp4"
        })

    except HTTPException as he:
        print("HTTPException in compare_image route:", he)
        raise he
    except Exception as e:
        print("Error in compare_image route:", e)
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Failed to process image."})

@router.post("/compare-image-multi-skeleton")
async def compare_image_multi_skeleton(
    image: UploadFile = File(...),
    s3_folders: List[str] = Form(...),
    sift_left: float = Form(20.0),
    sift_right: float = Form(20.0),
    sift_up: float = Form(20.0),
    sift_down: float = Form(20.0),
):
    """
    Draws all pose skeletons from the given S3 folders on each frame, overlapping.
    Most recent skeletons are drawn on top.
    """
    try:
        print("Starting compare_image_multi_skeleton route")
        os.makedirs("temp_uploads", exist_ok=True)
        temp_image_path = os.path.join("temp_uploads", "compare_image_multi.jpg")
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        print(f"Saved uploaded image: {temp_image_path}")

        # Collect all transformed poses first
        all_transformed_poses = []
        all_frame_numbers = set()
        
        for i, folder in enumerate(sorted(s3_folders)):  # sort so latest is last (drawn on top)
            key = folder.replace("s3://route-keypoints/", "").strip("/")
            print(f"Processing folder {i+1}/{len(s3_folders)}: {key}")
            
            # Load pose and SIFT data for this folder
            pose_data = load_pose_data_from_path(key)
            sift_kps, sift_descs = load_sift_data_from_path(key)
            
            # Convert pose data to the format expected
            formatted_pose_data = {}
            for frame, items in pose_data.items():
                if isinstance(items, list) and len(items) > 0:
                    skeleton = items[0]
                    if isinstance(skeleton, dict):
                        formatted_skeleton = {}
                        for joint_id, coords in skeleton.items():
                            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                                formatted_skeleton[joint_id] = {"x": float(coords[0]), "y": float(coords[1])}
                            elif isinstance(coords, dict) and "x" in coords and "y" in coords:
                                formatted_skeleton[joint_id] = {"x": float(coords["x"]), "y": float(coords["y"])}
                        formatted_pose_data[int(frame)] = formatted_skeleton
                elif isinstance(items, dict):
                    formatted_skeleton = {}
                    for joint_id, coords in items.items():
                        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                            formatted_skeleton[joint_id] = {"x": float(coords[0]), "y": float(coords[1])}
                        elif isinstance(coords, dict) and "x" in coords and "y" in coords:
                            formatted_skeleton[joint_id] = {"x": float(coords["x"]), "y": float(coords["y"])}
                    formatted_pose_data[int(frame)] = formatted_skeleton
            
            if not formatted_pose_data:
                print(f"No valid pose data found for {key}")
                continue
            
            # Transform this skeleton's poses to match the reference image
            transformed_poses = transform_poses_to_image(
                temp_image_path,
                formatted_pose_data,
                sift_kps,
                sift_descs,
                sift_left=sift_left,
                sift_right=sift_right,
                sift_up=sift_up,
                sift_down=sift_down
            )
            if not transformed_poses or len(transformed_poses) == 0:
                print(f"No valid transformed poses found for {key}")
                continue

            all_transformed_poses.append(transformed_poses)
            all_frame_numbers.update(transformed_poses.keys())
            print(f"Got {len(transformed_poses)} transformed frames for skeleton {i+1}")

        if not all_transformed_poses:
            raise HTTPException(500, "No valid transformed poses found")

        # Interpolate missing frames for each skeleton
        def interpolate_pose_dict(pose_dict, all_frames):
            interp = {}
            keys_sorted = sorted(pose_dict.keys())
            for idx in range(len(keys_sorted) - 1):
                f1, f2 = keys_sorted[idx], keys_sorted[idx + 1]
                interp[f1] = pose_dict[f1]
                gap = f2 - f1
                if gap > 1:
                    for j in range(1, gap):
                        alpha = j / gap
                        interp[f1 + j] = linear_interpolate_pose(pose_dict[f1], pose_dict[f2], alpha)
            interp[keys_sorted[-1]] = pose_dict[keys_sorted[-1]]
            # Fill in any missing frames (e.g., if frame numbers are not consecutive)
            min_frame = min(all_frames)
            max_frame = max(all_frames)
            for frame_num in range(min_frame, max_frame + 1):
                if frame_num not in interp:
                    prev = max([k for k in interp if k < frame_num], default=None)
                    nxt = min([k for k in interp if k > frame_num], default=None)
                    if prev is not None and nxt is not None:
                        alpha = (frame_num - prev) / (nxt - prev)
                        interp[frame_num] = linear_interpolate_pose(interp[prev], interp[nxt], alpha)
            return interp

        # Interpolate all skeletons to cover all frames
        all_frame_numbers = set()
        for poses in all_transformed_poses:
            all_frame_numbers.update(poses.keys())
        all_frame_numbers = sorted(all_frame_numbers)
        all_transformed_poses_interp = [interpolate_pose_dict(poses, all_frame_numbers) for poses in all_transformed_poses]

        # Create video by drawing all skeletons on each frame
        bg_img = cv2.imread(temp_image_path)
        height, width = bg_img.shape[:2]
        VIDEO_OUT_DIR = os.path.join("temp_uploads", "pose_feature_data", "output_video")
        os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
        out_path = os.path.join(VIDEO_OUT_DIR, "output_video_multi.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height))
        if not writer.isOpened():
            raise HTTPException(500, f"Failed to open video writer for {out_path}")

        # For each frame number, draw all available skeletons (now all have all frames)
        for frame_num in all_frame_numbers:
            frame = bg_img.copy()
            for transformed_poses in all_transformed_poses_interp:
                if frame_num in transformed_poses:
                    draw_pose(frame, transformed_poses[frame_num])
            writer.write(frame)
        writer.release()
        print(f"Created multi-skeleton video with {len(all_frame_numbers)} frames (all skeletons interpolated)")

        # Convert final video for browser
        browser_ready = os.path.join(VIDEO_OUT_DIR, "output_video_multi_browser.mp4")
        convert_video_for_browser(out_path, browser_ready)

        print("Returning multi-skeleton video response")
        return JSONResponse({
            "message": "Multi-skeleton video created successfully.",
            "video_url": "/static/pose_feature_data/output_video/output_video_multi_browser.mp4"
        })

    except HTTPException as he:
        print("HTTPException in compare_image_multi_skeleton route:", he)
        raise he
    except Exception as e:
        print("Error in compare_image_multi_skeleton route:", e)
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Failed to process multi-skeleton image."})

@router.post("/compare-image-multi-cropped")
async def compare_image_multi_cropped(
    image: UploadFile = File(...),
    s3_folders: List[str] = Form(...),
    sift_left: float = Form(20.0),
    sift_right: float = Form(20.0),
    sift_up: float = Form(20.0),
    sift_down: float = Form(20.0),
):
    """
    For each S3 folder, generate a video of the climber, then concatenate all videos side by side.
    """
    import subprocess
    print("Starting compare_image_multi_cropped route (side by side)")
    os.makedirs("temp_uploads", exist_ok=True)
    temp_image_path = os.path.join("temp_uploads", "compare_image_multi.jpg")
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    print(f"Saved uploaded image: {temp_image_path}")

    video_paths = []
    for i, folder in enumerate(sorted(s3_folders)):
        key = folder.replace("s3://route-keypoints/", "").strip("/")
        print(f"Processing folder {i+1}/{len(s3_folders)}: {key}")
        pose_data = load_pose_data_from_path(key)
        sift_kps, sift_descs = load_sift_data_from_path(key)
        formatted_pose_data = {}
        for frame, items in pose_data.items():
            if isinstance(items, list) and len(items) > 0:
                skeleton = items[0]
                if isinstance(skeleton, dict):
                    formatted_skeleton = {}
                    for joint_id, coords in skeleton.items():
                        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                            formatted_skeleton[joint_id] = {"x": float(coords[0]), "y": float(coords[1])}
                        elif isinstance(coords, dict) and "x" in coords and "y" in coords:
                            formatted_skeleton[joint_id] = {"x": float(coords["x"]), "y": float(coords["y"])}
                    formatted_pose_data[int(frame)] = formatted_skeleton
            elif isinstance(items, dict):
                formatted_skeleton = {}
                for joint_id, coords in items.items():
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        formatted_skeleton[joint_id] = {"x": float(coords[0]), "y": float(coords[1])}
                    elif isinstance(coords, dict) and "x" in coords and "y" in coords:
                        formatted_skeleton[joint_id] = {"x": float(coords["x"]), "y": float(coords["y"])}
                formatted_pose_data[int(frame)] = formatted_skeleton
        if not formatted_pose_data:
            print(f"No valid pose data found for {key}")
            continue
        print(f"formatted_pose_data keys for {key}: {list(formatted_pose_data.keys())}")
        print(f"sift_kps length: {len(sift_kps)}, sift_descs length: {len(sift_descs)}")
        # Generate video for this folder
        video_out = os.path.join(VIDEO_OUT_DIR, f"output_video_{i+1}.mp4")
        create_video_from_static_image(
            image_path=temp_image_path,
            pose_landmarks=formatted_pose_data,
            stored_keypoints_all=sift_kps,
            stored_descriptors_all=sift_descs,
            output_video=os.path.basename(video_out)
        )
        if not os.path.exists(video_out):
            print(f"Video file was not created: {video_out}")
        else:
            print(f"Video file created: {video_out}")
        video_paths.append(video_out)
    if not video_paths:
        raise HTTPException(500, "No valid videos generated for any folder")
    # Check that all video files exist before stacking
    missing = [vp for vp in video_paths if not os.path.exists(vp)]
    if missing:
        print(f"Missing video files before ffmpeg hstack: {missing}")
        raise HTTPException(500, f"Missing video files: {missing}")
    # Attach videos side by side using ffmpeg hstack
    if len(video_paths) == 1:
        concat_out = video_paths[0]
    else:
        hstack_filter = f"hstack=inputs={len(video_paths)}"
        concat_out = os.path.join(VIDEO_OUT_DIR, "output_video_multi_sidebyside.mp4")
        hstack_cmd = [
            "ffmpeg", "-y"
        ]
        for vp in video_paths:
            hstack_cmd += ["-i", vp]
        hstack_cmd += ["-filter_complex", hstack_filter, "-c:a", "copy", concat_out]
        print("Stacking videos side by side:", " ".join(hstack_cmd))
        subprocess.run(hstack_cmd, check=True)
    # Convert for browser
    browser_ready = os.path.join(VIDEO_OUT_DIR, "output_video_multi_sidebyside_browser.mp4")
    convert_video_for_browser(concat_out, browser_ready)
    print("Returning multi-sidebyside video response")
    return JSONResponse({
        "message": "Multi-sidebyside video created successfully.",
        "video_url": "/static/pose_feature_data/output_video/output_video_multi_sidebyside_browser.mp4"
    })

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
    """Transform pose landmarks to match the reference image using SIFT feature matching"""
    # This is the same transformation logic from create_video_from_static_image_streamed
    # but extracted to just return the transformed poses without creating a video
    
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
            continue

        transformed[frame_num] = apply_transform(T, pose_landmarks[frame_num])
        prev_T = T
        prev_query_indices = set(m.queryIdx for m in use_matches)

    return transformed
