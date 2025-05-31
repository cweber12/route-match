import os
import uuid
import json

TEMP_STORAGE_PATH = os.getenv("TEMP_STORAGE_PATH", os.path.join(os.getcwd(), "temp_uploads"))
os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)

def save_temp_file(file_bytes, file_ext=".mp4"):
    filename = f"temp_{uuid.uuid4()}{file_ext}"
    path = os.path.join(TEMP_STORAGE_PATH, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path

def save_temp_json(data):
    filename = f"pose_{uuid.uuid4()}.json"
    path = os.path.join(TEMP_STORAGE_PATH, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)  
    return path

def save_temp_sift_json(sift_data):
    filename = f"sift_{uuid.uuid4()}.json"
    path = os.path.join(TEMP_STORAGE_PATH, filename)
    with open(path, "w") as f:
        json.dump(sift_data, f, indent=2) 
    return path

def save_temp_video(video_bytes):
    filename = f"video_{uuid.uuid4()}.mp4"
    path = os.path.join(TEMP_STORAGE_PATH, filename)
    with open(path, "wb") as f:
        f.write(video_bytes)
    return path

def save_temp_preprocessing_log(preprocessing_log_detailed):

    file_id = f"preprocessing_{uuid.uuid4().hex}.json"
    path = os.path.join(TEMP_STORAGE_PATH, file_id)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(preprocessing_log_detailed, f, indent=2)

    print(f"Saved preprocessing log to: {path}")
    return path