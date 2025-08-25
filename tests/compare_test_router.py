from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid

router = APIRouter()


@router.post("/compare-image")
async def compare_image_test(
    image: UploadFile = File(None),
    built_in_image: str = Form("") ,
    s3_folders: list[str] = Form(...),
    sift_left: float = Form(20.0),
    sift_right: float = Form(20.0),
    sift_up: float = Form(20.0),
    sift_down: float = Form(20.0),
    line_color: str = Form("100,255,0"),
    point_color: str = Form("0,100,255"),
    fps: int = Form(24),
):
    # Minimal validation and stubbed behavior so tests can exercise the route
    if image is None and not built_in_image:
        raise HTTPException(400, "No image provided")

    # Create a placeholder output file so caller can validate existence
    out_dir = os.path.join("temp_uploads", "pose_feature_data", "output_video")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"output_video_{uuid.uuid4().hex}.mp4")
    with open(out_path, "wb") as f:
        f.write(b"dummy video content")

    return JSONResponse({
        "message": "test compare completed",
        "video_url": "/static/pose_feature_data/output_video/" + os.path.basename(out_path),
    })
