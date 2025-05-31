from fastapi import APIRouter
from fastapi.responses import JSONResponse
import shutil
import os

router = APIRouter()

@router.delete("/clear-temp")
async def clear_temp_folder():
    temp_path = os.path.join(os.getcwd(), "temp_uploads")
    try:
        if os.path.exists(temp_path):
            for filename in os.listdir(temp_path):
                file_path = os.path.join(temp_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        return JSONResponse({"message": "Temp folder cleared"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/clear-output")
async def clear_output_folder():
    output_dir = "static"
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        return JSONResponse({"message": "Output folder cleared."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})