
import os
import json
import cv2
import numpy as np
import boto3

S3_BUCKET = "route-keypoints"
LOCAL_STORAGE = "static/pose_feature_data"
POSE_JSON = os.path.join(LOCAL_STORAGE, "pose_landmarks.json")
SIFT_JSON = os.path.join(LOCAL_STORAGE, "sift_keypoints.json")

s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

def download_from_s3(s3_key, local_path):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"⬇Downloading from S3: {s3_key}")
        s3.download_file(S3_BUCKET, s3_key, local_path)

def find_matching_file(s3_folder, prefix):
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_folder)
    if "Contents" not in response:
        print(f"No contents found for prefix: {s3_folder}")
        return None
    for obj in response["Contents"]:
        filename = obj["Key"].split("/")[-1]
        if filename.startswith(prefix) and filename.endswith(".json"):
            print(f"Matched file: {filename}")
            return obj["Key"]
    return None

def load_pose_data_from_path(s3_folder):
    s3_key = find_matching_file(s3_folder, "pose_")
    if not s3_key:
        raise FileNotFoundError("Pose file not found in S3 folder.")
    download_from_s3(s3_key, POSE_JSON)
    with open(POSE_JSON) as f:
        data = json.load(f)
    poses = {}
    for item in data:
        if not isinstance(item, dict) or "frame" not in item:
            continue
        poses.setdefault(item["frame"], []).append(item)
    return poses

def load_sift_data_from_path(s3_folder):
    s3_key = find_matching_file(s3_folder, "sift_")
    if not s3_key:
        raise FileNotFoundError("SIFT file not found in S3 folder.")
    download_from_s3(s3_key, SIFT_JSON)
    with open(SIFT_JSON) as f:
        raw_data = json.load(f)
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
    return kps_all, descs_all
