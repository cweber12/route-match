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

def download_from_s3(s3_key, local_path):
    if os.path.exists(local_path):
        print(f"File already exists locally: {local_path}")
    else:
        print(f"File does not exist locally, proceeding to download: {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"⬇Downloading from S3: {s3_key}")
        s3.download_file(S3_BUCKET, s3_key, local_path)
        if os.path.getsize(local_path) == 0:
            raise ValueError(f"Downloaded file is empty: {local_path}")

def find_matching_file(s3_folder, prefix):
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_folder)
    print(f"S3 list_objects_v2 response: {response}")
    if "Contents" not in response:
        print(f"No contents found for prefix: {s3_folder}")
        return None
    for obj in response["Contents"]:
        filename = obj["Key"].split("/")[-1]
        print(f"Checking file: {filename}, S3 key: {obj['Key']}")
        if filename.startswith(prefix) and filename.endswith(".json"):
            print(f"Matched file: {filename}")
            return obj["Key"]
    return None

def load_pose_data_from_path(s3_folder):

    print("=== ENTERED load_pose_data_from_path ===")
    print(f"LOCAL_STORAGE: {LOCAL_STORAGE}")
    print(f"Files in LOCAL_STORAGE before cleanup: {os.listdir(LOCAL_STORAGE)}")

    # Clear old pose and SIFT files before downloading new ones
    old_files = [f for f in os.listdir(LOCAL_STORAGE) if f.startswith("pose_landmarks") or f.startswith("sift_keypoints")]
    print(f"Found old files to delete: {old_files}")
    for old_file in old_files:
        try:
            file_path = os.path.join(LOCAL_STORAGE, old_file)
            os.remove(file_path)
            print(f"Deleted old file: {file_path}")
        except Exception as e:
            print(f"Failed to delete old file {file_path}: {e}")

    print(f"Files in LOCAL_STORAGE after cleanup: {os.listdir(LOCAL_STORAGE)}")

    pose_json = os.path.join(LOCAL_STORAGE, f"pose_landmarks_{int(time.time())}_{os.urandom(4).hex()}.json")
    print(f"Generated unique local file path for pose JSON: {pose_json}")
    s3_key = find_matching_file(s3_folder, "pose_")
    if not s3_key:
        raise FileNotFoundError("Pose file not found in S3 folder.")
    print(f"Matched S3 key for pose JSON: {s3_key}")
    download_from_s3(s3_key, pose_json)

    # Ensure downloaded file is not empty
    if os.path.getsize(pose_json) == 0:
        raise ValueError(f"Downloaded pose file is empty: {pose_json}")

    with open(pose_json, "r") as f:
        try:
            data = json.load(f)
            print(f"Downloaded pose file content: {data}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON content from {pose_json}: {e}")

    poses = {}
    for item in data:
        if not isinstance(item, dict) or "frame" not in item:
            continue
        poses.setdefault(item["frame"], []).append(item)
    print(f"Loaded pose data: {len(poses)} frames")
    return poses

def load_sift_data_from_path(s3_folder):

    print("=== ENTERED load_sift_data_from_path ===")
    print(f"LOCAL_STORAGE: {LOCAL_STORAGE}")
    print(f"Files in LOCAL_STORAGE before cleanup: {os.listdir(LOCAL_STORAGE)}")

    # Clear old pose and SIFT files before downloading new ones
    old_files = [f for f in os.listdir(LOCAL_STORAGE) if f.startswith("pose_landmarks") or f.startswith("sift_keypoints")]
    print(f"Found old files to delete: {old_files}")
    for old_file in old_files:
        try:
            file_path = os.path.join(LOCAL_STORAGE, old_file)
            os.remove(file_path)
            print(f"Deleted old file: {file_path}")
        except Exception as e:
            print(f"Failed to delete old file {file_path}: {e}")

    print(f"Files in LOCAL_STORAGE after cleanup: {os.listdir(LOCAL_STORAGE)}")

    sift_json = os.path.join(LOCAL_STORAGE, f"sift_keypoints_{int(time.time())}_{os.urandom(4).hex()}.json")
    print(f"Generated unique local file path for SIFT JSON: {sift_json}")
    s3_key = find_matching_file(s3_folder, "sift_")
    if not s3_key:
        raise FileNotFoundError("SIFT file not found in S3 folder.")
    print(f"Matched S3 key for SIFT JSON: {s3_key}")
    download_from_s3(s3_key, sift_json)

    # Ensure downloaded file is not empty
    if os.path.getsize(sift_json) == 0:
        raise ValueError(f"Downloaded SIFT file is empty: {sift_json}")

    with open(sift_json, "r") as f:
        try:
            raw_data = json.load(f)
            print(f"Downloaded SIFT file content: {raw_data}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON content from {sift_json}: {e}")

    frame_dict = {}
    for entry in raw_data:
        frame = entry["frame"]
        frame_dict.setdefault(frame, []).append(entry)
    kps_all, descs_all = [], []
    for frame in sorted(frame_dict):
        frame_data = frame_dict[frame]
        kps = [cv2.KeyPoint(x=d["x"], y=d["y"], size=d["size"]) for d in frame_data]
        desc = [d["descriptor"] for d in frame_data]
        kps_all.append(kps)
        descs_all.append(np.array(desc, dtype=np.float32))
    print(f"Loaded SIFT data: {len(kps_all)} keypoints sets, {len(descs_all)} descriptors sets")
    return kps_all, descs_all
