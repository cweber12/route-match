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
    for file in os.listdir(VIDEO_OUT_DIR):
        if file.endswith((".mp4", ".avi")):
            os.remove(os.path.join(VIDEO_OUT_DIR, file))
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
    print("Static SIFT mode:", static_mode)

    transformed = {}
    prev_T = None
    prev_query_indices = set()
    skipped_frames = 0

    frame_keys = sorted(pose_landmarks.keys())
    for i, frame_num in enumerate(frame_keys):
        if static_mode:
            kp = stored_keypoints_all[0]
            desc = stored_descriptors_all[0]
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

        transformed[frame_num] = apply_transform(T, pose_landmarks[frame_num])
        prev_T = T
        prev_query_indices = set(m.queryIdx for m in use_matches)

    if len(transformed) < 2:
        print("Not enough transformed frames to generate video")
        return

    write_pose_video(out_path, ref_img, transformed)

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
    output_video="output_video.mp4"
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

    frame_keys = sorted([int(k) for k in pose_landmarks.keys()])
    frame_keys_str = [str(k) for k in frame_keys]
    height, width = ref_img.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height))
    if not writer.isOpened():
        print(f"Failed to open video writer for {out_path}")
        return

    prev_T = None
    prev_query_indices = set()
    skipped_frames = 0

    i = 0
    while i < len(frame_keys):
        frame_num = frame_keys[i]
        frame_num_str = str(frame_num)
        # Get SIFT data for this frame
        if len(stored_keypoints_all) == 1:
            kp = stored_keypoints_all[0]
            desc = stored_descriptors_all[0]
        elif i < len(stored_keypoints_all):
            kp = stored_keypoints_all[i]
            desc = stored_descriptors_all[i]
        else:
            print(f"⚠ No matching SIFT keypoints for frame {frame_num}")
            i += 1
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
            i += 1
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
            i += 1
            continue

        # Transform and draw for this frame only
        if frame_num not in pose_landmarks:
            print(f"Frame {frame_num} not found in pose_landmarks keys: {list(pose_landmarks.keys())}")
            i += 1
            continue
        transformed_pose = apply_transform(T, pose_landmarks[frame_num])
        frame = ref_img.copy()
        draw_pose(frame, transformed_pose)
        writer.write(frame)

        # Interpolation logic
        # Look ahead to next valid frame
        j = i + 1
        while j < len(frame_keys):
            next_frame_num = frame_keys[j]
            next_frame_num_str = str(next_frame_num)
            # Try to get SIFT and pose for next frame
            if len(stored_keypoints_all) == 1:
                next_kp = stored_keypoints_all[0]
                next_desc = stored_descriptors_all[0]
            elif j < len(stored_keypoints_all):
                next_kp = stored_keypoints_all[j]
                next_desc = stored_descriptors_all[j]
            else:
                j += 1
                continue

            next_matches = match_features(
                next_desc, ref_desc,
                prev_query_indices=prev_query_indices,
                min_shared_matches=0,
                ratio_thresh=0.75,
                distance_thresh=300,
                min_required_matches=10,
                top_n=150,
                debug=True
            )

            if not next_matches:
                j += 1
                continue

            next_T = compute_affine_transform(
                next_kp, ref_kp, next_matches,
                prev_T=prev_T,
                alpha=0.9,
                debug=True
            )

            if next_T is None:
                j += 1
                continue

            # If there is a gap, interpolate
            gap = next_frame_num - frame_num
            if gap > 1:
                for interp_idx in range(1, gap):
                    alpha = interp_idx / gap
                    interp_pose = linear_interpolate_pose(
                        apply_transform(T, pose_landmarks[frame_num]),
                        apply_transform(next_T, pose_landmarks[next_frame_num]),
                        alpha
                    )
                    interp_frame = ref_img.copy()
                    draw_pose(interp_frame, interp_pose)
                    writer.write(interp_frame)
            break  # Only interpolate to the next valid frame
        prev_T = T
        prev_query_indices = set(m.queryIdx for m in use_matches)
        i += 1

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
