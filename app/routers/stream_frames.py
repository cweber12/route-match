from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import json
import io
import typing as t

router = APIRouter()


def _normalize_landmarks(frame_entry: t.Any) -> dict:
    """Normalize many possible landmark formats into a dict idx->(x,y)."""
    if frame_entry is None:
        return {}
    # dict keyed by index -> (x,y) or {'x':..,'y':..}
    if isinstance(frame_entry, dict):
        # check if keys look like integers
        try:
            return {int(k): (v[0], v[1]) if isinstance(v, (list, tuple)) else (v.get("x"), v.get("y")) for k, v in frame_entry.items() if v is not None}
        except Exception:
            pass

    # list of landmark dicts
    if isinstance(frame_entry, (list, tuple)):
        out = {}
        for lm in frame_entry:
            if not isinstance(lm, dict):
                continue
            idx = lm.get("landmark_number") or lm.get("index") or lm.get("id")
            if idx is None:
                continue
            if "x" in lm and "y" in lm:
                try:
                    out[int(idx)] = (float(lm["x"]), float(lm["y"]))
                except Exception:
                    continue
        return out

    return {}


@router.post("/stream-frames")
async def stream_frames(image: UploadFile = File(...), pose_file: UploadFile = File(...)):
    """Stream one frame per stored pose entry with the skeleton drawn on the reference image.

    Accepts:
    - image: the reference image (JPEG/PNG)
    - pose_file: JSON file mapping frame keys to landmarks

    The response is multipart/x-mixed-replace of JPEG frames (one per pose frame).
    """
    # Read pose JSON
    try:
        pose_bytes = await pose_file.read()
        pose_data = json.loads(pose_bytes.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid pose JSON: {e}")

    # Read image bytes
    try:
        img_bytes = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {e}")

    # Defer heavy imports until we actually need them
    try:
        import cv2
        import numpy as np
        from app.services.draw_points import draw_pose
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server missing imaging libs: {e}")

    # Decode image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")

    # Prepare ordered frame keys
    try:
        frame_keys = sorted([int(k) for k in pose_data.keys()])
    except Exception:
        # fallback: if pose_data is list-like use indices
        if isinstance(pose_data, (list, tuple)):
            frame_keys = list(range(len(pose_data)))
        else:
            frame_keys = sorted([int(k) for k in pose_data.keys()])

    boundary = "frame"

    def gen():
        for fk in frame_keys:
            entry = pose_data.get(str(fk)) if str(fk) in pose_data else pose_data.get(fk)
            landmarks = _normalize_landmarks(entry)
            # copy the image for drawing
            frame = img.copy()
            try:
                draw_pose(frame, landmarks)
            except Exception:
                # if draw fails, continue with original frame
                pass

            # encode to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            data = jpeg.tobytes()
            yield (b"--%b\r\n" % boundary.encode())
            yield b"Content-Type: image/jpeg\r\n"
            yield (b"Content-Length: %d\r\n\r\n" % len(data))
            yield data
            yield b"\r\n"

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")