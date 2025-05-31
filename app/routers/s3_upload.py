from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from datetime import datetime
import boto3
import os
import json
import shutil
from pathlib import Path

router = APIRouter()

s3_client = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# Create temp_uploads directory if it doesn't exist
os.makedirs("temp_uploads", exist_ok=True)

@router.post("/upload-json")
async def upload_pose_and_sift_to_s3(
    pose_file_path: str = Form(...),
    sift_file_path: str = Form(...),
    user_name: str = Form(...),
    location: str = Form(...),
    route_name: str = Form(...),
    coordinates: str = Form(None),  # passed from frontend as JSON string
    location_image: UploadFile = File(None),
    route_image: UploadFile = File(None),
    attempt_image: UploadFile = File(None),
    route_description: str = Form(None),
):

    missing = []
    if not os.path.exists(pose_file_path):
        missing.append("Pose JSON")
    if not os.path.exists(sift_file_path):
        missing.append("SIFT JSON")
    if missing:
        raise HTTPException(status_code=404, detail=", ".join(missing) + " file(s) not found")

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = f"{user_name}/{location}/{route_name}/{timestamp}"
        bucket_name = os.getenv("S3_BUCKET_NAME", "route-keypoints")

        # Upload JSON files
        pose_key = f"{base_path}/{os.path.basename(pose_file_path)}"
        sift_key = f"{base_path}/{os.path.basename(sift_file_path)}"
        s3_client.upload_file(pose_file_path, bucket_name, pose_key)
        s3_client.upload_file(sift_file_path, bucket_name, sift_key)

        # Upload route_description.json if provided
        description_key = None
        if route_description:
            temp_desc_path = "temp_uploads/route_description.json"
            with open(temp_desc_path, "w") as f:
                json.dump({"description": route_description}, f)

            description_key = f"{base_path}/route_description.json"
            s3_client.upload_file(temp_desc_path, bucket_name, description_key)
            os.remove(temp_desc_path)

        # Upload images
        image_keys = {}

        async def upload_image(file: UploadFile, label: str):
            ext = os.path.splitext(file.filename)[-1]
            image_key = f"{base_path}/{label}_image{ext}"
            temp_path = f"temp_uploads/{label}_{file.filename}"
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            # Removed ACL since ACLs are disabled in bucket
            s3_client.upload_file(temp_path, bucket_name, image_key)
            os.remove(temp_path)
            image_keys[label] = image_key

        if location_image:
            await upload_image(location_image, "location")
        if route_image:
            await upload_image(route_image, "route")
        if attempt_image:
            await upload_image(attempt_image, "attempt")

        # Upload coordinates.json
        coords_key = None
        if coordinates:
            coords_dict = json.loads(coordinates)
            temp_coords_path = "temp_uploads/coordinates.json"
            with open(temp_coords_path, "w") as f:
                json.dump(coords_dict, f)

            coords_key = f"{user_name}/{location}/gps_location/coordinates.json"
            s3_client.upload_file(temp_coords_path, bucket_name, coords_key)
            os.remove(temp_coords_path)

        return {
            "message": "Upload successful",
            "pose_s3_key": pose_key,
            "sift_s3_key": sift_key,
            "coordinates_s3_key": coords_key,
            "description_s3_key": description_key,
            **{f"{k}_image_s3_key": v for k, v in image_keys.items()}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")
