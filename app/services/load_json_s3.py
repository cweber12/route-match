# load_json_s3.py
# ----------------------------------------------------------------------------------------------------------
# This module provides functions to load pose and SIFT keypoint data from JSON files stored in an S3 bucket.
# It includes functions to download files from S3, parse JSON content, and validate the data.
# ----------------------------------------------------------------------------------------------------------

import os
import json 
import cv2
import numpy as np
import boto3
import time

S3_BUCKET = "route-keypoints"
LOCAL_STORAGE = "static/pose_feature_data"

s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# Call this function ONCE before any S3 downloads to clean up local storage
def cleanup_local_storage():
    try:
        files_before = os.listdir(LOCAL_STORAGE)
    except Exception as e:
        print(f"[cleanup_local_storage] Exception listing LOCAL_STORAGE: {e}")
        files_before = []
    old_files = [f for f in files_before if f.startswith("pose_landmarks") or f.startswith("sift_keypoints")]
    for old_file in old_files:
        try:
            file_path = os.path.join(LOCAL_STORAGE, old_file)
            os.remove(file_path)
        except Exception as e:
            print(f"[cleanup_local_storage] Failed to delete old file {file_path}: {e}")
    
def download_from_s3(s3_key, local_path):
    if os.path.exists(local_path):
        print(f"[download_from_s3] File already exists locally: {local_path}")
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            s3.download_file(S3_BUCKET, s3_key, local_path)
        except Exception as e:
            print(f"[download_from_s3] Exception during download: {e}")
            raise
        if os.path.getsize(local_path) == 0:
            raise ValueError(f"Downloaded file is empty: {local_path}")

def find_matching_file(s3_folder, prefix):
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_folder)
    except Exception as e:
        print(f"[find_matching_file] Exception during S3 list_objects_v2: {e}")
        return None
    if "Contents" not in response:
        print(f"[find_matching_file] No contents found for prefix: {s3_folder}")
        return None
    for obj in response["Contents"]:
        filename = obj["Key"].split("/")[-1]
        if filename.startswith(prefix) and filename.endswith(".json"):
            return obj["Key"]
    return None

def load_pose_data_from_path(s3_folder):

    pose_json = os.path.join(LOCAL_STORAGE, f"pose_landmarks_{int(time.time())}_{os.urandom(4).hex()}.json")
    s3_key = find_matching_file(s3_folder, "pose_")
    if not s3_key:
        raise FileNotFoundError("Pose file not found in S3 folder.")
    download_from_s3(s3_key, pose_json)

    # Ensure downloaded file is not empty
    if os.path.getsize(pose_json) == 0:
        raise ValueError(f"Downloaded pose file is empty: {pose_json}")

    with open(pose_json, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON content from {pose_json}: {e}")

    poses = {}
    for item in data:
        if not isinstance(item, dict) or "frame" not in item:
            continue
        poses.setdefault(item["frame"], []).append(item)
    return poses

def load_sift_data_from_path(s3_folder):
    sift_json = os.path.join(LOCAL_STORAGE, f"sift_keypoints_{int(time.time())}_{os.urandom(4).hex()}.json")
    s3_key = find_matching_file(s3_folder, "sift_")
    if not s3_key:
        raise FileNotFoundError("SIFT file not found in S3 folder.")
    
    print(f"[download_from_s3] Called with s3_key={s3_key}, local_path={sift_json}")
    download_from_s3(s3_key, sift_json)

    if os.path.getsize(sift_json) == 0:
        raise ValueError(f"Downloaded SIFT file is empty: {sift_json}")

    with open(sift_json, "r") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON content from {sift_json}: {e}")

    frame_dimensions = None
    
    # Handle both old and new JSON formats
    if isinstance(raw_data, dict) and "sift_features" in raw_data:
        # New format: {"frame_dimensions": {...}, "sift_features": [...]}
        print(f"[load_sift_data_from_path] Detected new JSON format with frame_dimensions")
        frame_dimensions = raw_data.get("frame_dimensions", {})
        features_data = raw_data["sift_features"]
        print(f"[load_sift_data_from_path] Frame dimensions: {frame_dimensions}")
    elif isinstance(raw_data, list):
        # Old format: [{"frame": ..., "x": ..., ...}, ...]
        print(f"[load_sift_data_from_path] Detected old JSON format (array)")
        features_data = raw_data
    else:
        raise ValueError(f"Unrecognized JSON format in {sift_json}")

    # Group features by frame
    frame_dict = {}
    for entry in features_data:
        frame = entry["frame"]
        frame_dict.setdefault(frame, []).append(entry)
    
    # Check if this is single-frame or multi-frame data
    unique_frames = list(frame_dict.keys())
    is_multi_frame = len(unique_frames) > 1
    
    print(f"[load_sift_data_from_path] Found {len(unique_frames)} unique frames: {unique_frames}")
    print(f"[load_sift_data_from_path] Multi-frame mode: {is_multi_frame}")
    
    # Convert to keypoints and descriptors
    if is_multi_frame:
        # Multi-frame: return frame-indexed data
        frame_data = {}
        for frame in sorted(frame_dict):
            frame_features = frame_dict[frame]
            kps = [cv2.KeyPoint(x=d["x"], y=d["y"], size=d["size"]) for d in frame_features]
            desc = np.array([d["descriptor"] for d in frame_features], dtype=np.float32)
            frame_data[frame] = {"keypoints": kps, "descriptors": desc}
        
        print(f"[load_sift_data_from_path] Returning multi-frame data for {len(frame_data)} frames")
        return frame_data, frame_dimensions, True  # True indicates multi-frame
    else:
        # Single-frame: return legacy format for backward compatibility
        kps_all, descs_all = [], []
        for frame in sorted(frame_dict):
            frame_features = frame_dict[frame]
            kps = [cv2.KeyPoint(x=d["x"], y=d["y"], size=d["size"]) for d in frame_features]
            desc = np.array([d["descriptor"] for d in frame_features], dtype=np.float32)
            kps_all.append(kps)
            descs_all.append(desc)
        
        print(f"[load_sift_data_from_path] Returning single-frame data")
        return (kps_all, descs_all), frame_dimensions, False  # False indicates single-frame
