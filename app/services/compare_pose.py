import os
import cv2
import numpy as np
from pathlib import Path
import subprocess
import json
import shutil

from .load_json_s3 import load_pose_data_from_path, load_sift_data_from_path
from .detect_img_sift import detect_sift
from .match_features import match_features, compute_affine_transform
from .draw_points import apply_transform, draw_pose
from .video_writer import write_pose_video

VIDEO_OUT_DIR = os.path.join("temp_uploads", "pose_feature_data", "output_video")
POSE_JSON = os.path.join("static", "pose_feature_data", "pose_landmarks.json")
SIFT_JSON = os.path.join("static", "pose_feature_data", "sift_keypoints.json")


def linear_interpolate_pose(pose1, pose2, alpha):
    # Only interpolate keys present in both poses
    shared_keys = set(pose1.keys()) & set(pose2.keys())
    result = {}
    for k in shared_keys:
        v1, v2 = pose1[k], pose2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            result[k] = linear_interpolate_pose(v1, v2, alpha)
        else:
            result[k] = (1 - alpha) * np.array(v1) + alpha * np.array(v2)
    return result


def clear_output_dir():
    for file in os.listdir(VIDEO_OUT_DIR):
        file_path = os.path.join(VIDEO_OUT_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def create_video_from_static_image(
    image_path,
    pose_landmarks,
    stored_keypoints_all,
    stored_descriptors_all,
    output_video="output_video.mp4"
):
    print("Starting video generation")
    ref_img = cv2.imread(image_path)
    if ref_img is None:
        print("Image not found:", image_path)
        return

    os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
    clear_output_dir()
    out_path = os.path.join(VIDEO_OUT_DIR, output_video)

    sift_config = {
        "nfeatures": 2000,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6
    }

    ref_kp, ref_desc = detect_sift(
        ref_img,
        sift_config=sift_config,
        use_clahe=False
    )

    with open(os.path.join(VIDEO_OUT_DIR, "sift_config.json"), "w") as f:
        json.dump(sift_config, f, indent=4)

    # Check if only one frame of SIFT data exists
    static_mode = len(stored_keypoints_all) == 1

    transformed = {}
    prev_T = None
    prev_query_indices = set()
    skipped_frames = 0
    # static run flag ensures matching only once for static SIFT mode
    static_run = True

    
    frame_keys = sorted(pose_landmarks.keys())
    # If static_mode, match and compute transform ONCE, then apply to all frames
    if static_mode:
  
        kp = stored_keypoints_all[0]
        desc = stored_descriptors_all[0]
        matches = match_features(
            desc, ref_desc,
            prev_query_indices=None,
            min_shared_matches=0,
            ratio_thresh=0.75,
            distance_thresh=300,
            min_required_matches=5,
            top_n=150,
            debug=True
        )
        if not matches:
            print("No good matches found for static SIFT mode.")
            return
        # Always use all matches for affine transform in static mode
        T = compute_affine_transform(
            kp, ref_kp, matches,
            prev_T=None,
            alpha=0.0,  # No smoothing for static
            debug=True
        )
        if T is None:
            print("No transform found for static SIFT mode.")
            return
        # Build transformed dict
        for frame_num in frame_keys:
            transformed[frame_num] = apply_transform(T, pose_landmarks[frame_num])
        # Interpolate between frames if there are gaps
        interp_transformed = {}
        frame_keys_sorted = sorted(transformed.keys())
        for idx in range(len(frame_keys_sorted) - 1):
            f1, f2 = frame_keys_sorted[idx], frame_keys_sorted[idx + 1]
            interp_transformed[f1] = transformed[f1]
            gap = f2 - f1
            if gap > 1:
                for j in range(1, gap):
                    alpha = j / gap
                    interp_transformed[f1 + j] = linear_interpolate_pose(transformed[f1], transformed[f2], alpha)
        interp_transformed[frame_keys_sorted[-1]] = transformed[frame_keys_sorted[-1]]
        # Fill in any missing frames (e.g., if frame numbers are not consecutive)
        min_frame = frame_keys_sorted[0]
        max_frame = frame_keys_sorted[-1]
        for frame_num in range(min_frame, max_frame + 1):
            if frame_num not in interp_transformed:
                # If a frame is missing (e.g., skipped), interpolate between nearest previous and next
                prev = max([k for k in interp_transformed if k < frame_num], default=None)
                nxt = min([k for k in interp_transformed if k > frame_num], default=None)
                if prev is not None and nxt is not None:
                    alpha = (frame_num - prev) / (nxt - prev)
                    interp_transformed[frame_num] = linear_interpolate_pose(interp_transformed[prev], interp_transformed[nxt], alpha)
        # Sort by frame number
        interp_transformed = dict(sorted(interp_transformed.items()))
        # Write video and cleanup
        write_pose_video(out_path, ref_img, interp_transformed)
        if os.path.exists(POSE_JSON): os.remove(POSE_JSON)
        if os.path.exists(SIFT_JSON): os.remove(SIFT_JSON)
        print(f"Finished writing video to {out_path}")
        print(f"Transformed frames: {len(interp_transformed)} (including interpolated)")
        return

    # ...existing code for multi-frame SIFT...
    for i, frame_num in enumerate(frame_keys):
        
        if static_run: 
            if static_mode:
                print("static-mode SIFT run")
                kp = stored_keypoints_all[0]
                desc = stored_descriptors_all[0]
                static_run = False
            elif i < len(stored_keypoints_all):
                kp = stored_keypoints_all[i]
                desc = stored_descriptors_all[i]
            else:
                print(f"⚠ No matching SIFT keypoints for frame {frame_num}")
                continue


            matches = match_features(
                desc, ref_desc,
                prev_query_indices=prev_query_indices,
                min_shared_matches=0,
                ratio_thresh=0.75,
                distance_thresh=300,
                min_required_matches=5,
                top_n=150,
                debug=True
            )

            if not matches:
                skipped_frames += 1
                continue

            shared = [m for m in matches if m.queryIdx in prev_query_indices] if prev_query_indices else matches
            use_matches = shared if len(shared) >= 5 else matches

        T = compute_affine_transform(
            kp, ref_kp, use_matches,
            prev_T=prev_T,
            alpha=0.9,
            debug=True
        )

        if T is None:
            skipped_frames += 1
            continue

        transformed[frame_num] = apply_transform(T, pose_landmarks[frame_num])
        prev_T = T
        prev_query_indices = set(m.queryIdx for m in use_matches)

    if len(transformed) < 2:
        print("Not enough transformed frames to generate video")
        return

    # Interpolate between frames if there are gaps
    interp_transformed = {}
    frame_keys_sorted = sorted(transformed.keys())
    for idx in range(len(frame_keys_sorted) - 1):
        f1, f2 = frame_keys_sorted[idx], frame_keys_sorted[idx + 1]
        interp_transformed[f1] = transformed[f1]
        gap = f2 - f1
        if gap > 1:
            for j in range(1, gap):
                alpha = j / gap
                interp_transformed[f1 + j] = linear_interpolate_pose(transformed[f1], transformed[f2], alpha)
    interp_transformed[frame_keys_sorted[-1]] = transformed[frame_keys_sorted[-1]]
    # Fill in any missing frames (e.g., if frame numbers are not consecutive)
    min_frame = frame_keys_sorted[0]
    max_frame = frame_keys_sorted[-1]
    for frame_num in range(min_frame, max_frame + 1):
        if frame_num not in interp_transformed:
            # If a frame is missing (e.g., skipped), interpolate between nearest previous and next
            prev = max([k for k in interp_transformed if k < frame_num], default=None)
            nxt = min([k for k in interp_transformed if k > frame_num], default=None)
            if prev is not None and nxt is not None:
                alpha = (frame_num - prev) / (nxt - prev)
                interp_transformed[frame_num] = linear_interpolate_pose(interp_transformed[prev], interp_transformed[nxt], alpha)
    # Sort by frame number
    interp_transformed = dict(sorted(interp_transformed.items()))
    # Write video and cleanup
    write_pose_video(out_path, ref_img, interp_transformed)

    if os.path.exists(POSE_JSON): os.remove(POSE_JSON)
    if os.path.exists(SIFT_JSON): os.remove(SIFT_JSON)

    print(f"Finished writing video to {out_path}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Transformed frames: {len(transformed)}")


def create_video_from_static_image_streamed(
    image_path,
    pose_landmarks,
    stored_keypoints_all,
    stored_descriptors_all,
    output_video="output_video.mp4",
    sift_left=20.0,
    sift_right=20.0,
    sift_up=20.0,
    sift_down=20.0,
    line_color=(100,255,0),
    point_color=(0,100,255)
):
    print("Starting streamed video generation")
    ref_img = cv2.imread(image_path)
    if ref_img is None:
        print("Image not found:", image_path)
        return

    os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
    for file in os.listdir(VIDEO_OUT_DIR):
        if file.endswith((".mp4", ".avi")):
            os.remove(os.path.join(VIDEO_OUT_DIR, file))
    out_path = os.path.join(VIDEO_OUT_DIR, output_video)

    # Calculate SIFT bounding box from parameters
    h_full, w_full = ref_img.shape[:2]
    x1_s = int(sift_left / 100 * w_full)
    y1_s = int(sift_up / 100 * h_full)
    x2_s = int(w_full - (sift_right / 100) * w_full)
    y2_s = int(h_full - (sift_down / 100) * h_full)
    bbox = (x1_s, y1_s, x2_s, y2_s)

    sift_config = {
        "nfeatures": 2000,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6
    }

    # Run SIFT on the cropped region, but transform keypoints to full-frame coordinates
    ref_img_cropped = ref_img[y1_s:y2_s, x1_s:x2_s]
    ref_kp_cropped, ref_desc = detect_sift(
        ref_img_cropped,
        sift_config=sift_config,
        use_clahe=False,
        bbox=None
    )
    # Transform cropped keypoints to full-frame coordinates
    ref_kp = []
    for kp in ref_kp_cropped:
        kp_full = cv2.KeyPoint(
            kp.pt[0] + x1_s,
            kp.pt[1] + y1_s,
            kp.size, kp.angle, kp.response, kp.octave, kp.class_id
        )
        ref_kp.append(kp_full)

    frame_keys = sorted([int(k) for k in pose_landmarks.keys()])
    height, width = ref_img.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height))
    if not writer.isOpened():
        print(f"Failed to open video writer for {out_path}")
        return

    prev_T = None
    prev_query_indices = set()
    skipped_frames = 0

    # Optimization: If only one set of SIFT keypoints/descriptors, match once and reuse
    if len(stored_keypoints_all) == 1:
        kp = stored_keypoints_all[0]
        desc = stored_descriptors_all[0]
        matches = match_features(
            desc, ref_desc,
            prev_query_indices=None,
            min_shared_matches=0,
            ratio_thresh=0.75,
            distance_thresh=300,
            min_required_matches=10,
            top_n=150,
            debug=True
        )
        if not matches:
            print("No good matches found for static SIFT mode.")
            return
        T = compute_affine_transform(
            kp, ref_kp, matches,
            prev_T=None,
            alpha=0.9,
            debug=True
        )
        if T is None:
            print("No transform found for static SIFT mode.")
            return
        # Build transformed dict for interpolation keys only (not all frames)
        transformed = {}
        for frame_num in frame_keys:
            transformed[frame_num] = apply_transform(T, pose_landmarks[frame_num])
        # Interpolate between frames and write directly to video
        frame_keys_sorted = sorted(transformed.keys())
        min_frame = frame_keys_sorted[0]
        max_frame = frame_keys_sorted[-1]
        for idx in range(len(frame_keys_sorted) - 1):
            f1, f2 = frame_keys_sorted[idx], frame_keys_sorted[idx + 1]
            pose1, pose2 = transformed[f1], transformed[f2]
            # Write f1
            frame = ref_img.copy()
            draw_pose(frame, pose1, line_color=line_color, point_color=point_color)
            writer.write(frame)
            gap = f2 - f1
            if gap > 1:
                for j in range(1, gap):
                    alpha = j / gap
                    interp_pose = linear_interpolate_pose(pose1, pose2, alpha)
                    frame_interp = ref_img.copy()
                    draw_pose(frame_interp, interp_pose, line_color=line_color, point_color=point_color)
                    writer.write(frame_interp)
        # Write last frame
        frame = ref_img.copy()
        draw_pose(frame, transformed[frame_keys_sorted[-1]], line_color=line_color, point_color=point_color)
        writer.write(frame)
        writer.release()
        print(f"Finished writing video to {out_path}")
        print(f"Processed frames: {max_frame - min_frame + 1}")
        return

    # For memory efficiency, process and write each frame (and interpolated) as you go
    transformed = {}
    prev_T = None
    prev_query_indices = set()
    prev_frame_num = None
    prev_pose = None
    for i, frame_num in enumerate(frame_keys):
        if i < len(stored_keypoints_all):
            kp = stored_keypoints_all[i]
            desc = stored_descriptors_all[i]
        else:
            print(f"⚠ No matching SIFT keypoints for frame {frame_num}")
            skipped_frames += 1
            continue
        matches = match_features(
            desc, ref_desc,
            prev_query_indices=prev_query_indices,
            min_shared_matches=0,
            ratio_thresh=0.75,
            distance_thresh=300,
            min_required_matches=10,
            top_n=150,
            debug=True
        )
        if not matches:
            skipped_frames += 1
            continue
        shared = [m for m in matches if m.queryIdx in prev_query_indices] if prev_query_indices else matches
        use_matches = shared if len(shared) >= 5 else matches
        T = compute_affine_transform(
            kp, ref_kp, use_matches,
            prev_T=prev_T,
            alpha=0.9,
            debug=True
        )
        if T is None:
            skipped_frames += 1
            continue
        pose = apply_transform(T, pose_landmarks[frame_num])
        # Interpolate and write between previous and current frame
        if prev_frame_num is not None and prev_pose is not None:
            gap = frame_num - prev_frame_num
            if gap > 1:
                for j in range(1, gap):
                    alpha = j / gap
                    interp_pose = linear_interpolate_pose(prev_pose, pose, alpha)
                    frame_interp = ref_img.copy()
                    draw_pose(frame_interp, interp_pose, line_color=line_color, point_color=point_color)
                    writer.write(frame_interp)
        # Write current frame
        frame = ref_img.copy()
        draw_pose(frame, pose, line_color=line_color, point_color=point_color)
        writer.write(frame)
        prev_frame_num = frame_num
        prev_pose = pose
        prev_T = T
        prev_query_indices = set(m.queryIdx for m in use_matches)
    writer.release()
    print(f"Finished writing video to {out_path}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Processed frames: {len(frame_keys) - skipped_frames}")


import os
import subprocess

def convert_video_for_browser(input_path: str, output_path: str) -> None:
 
    # 1) verify the raw video exists
    if not os.path.exists(input_path):
        print(f"Cannot convert video — file not found at {input_path}")
        return

    # 2) find ffmpeg on PATH
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        print("ffmpeg executable not found in your PATH. Please install ffmpeg.")
        return

    # 3) build the command
    cmd = [
        ffmpeg_exe,
        "-y",                # overwrite output if it exists
        "-i", input_path,    # input file
        "-c:v", "libx264",   # encode video with H.264
        "-crf", "23",        # quality level
        "-preset", "veryfast",
        "-movflags", "+faststart",  # optimize for web
        output_path
    ]

    print("Converting with FFmpeg:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully created browser‑friendly video at {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed (exit {e.returncode}): {e}")
