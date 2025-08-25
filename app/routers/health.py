# app/routers/health.py
# ------------------------------------------------------------------------
# Health check endpoint to verify file system access and other basic checks.
# For debugging and ensuring the application can access necessary resources.
# ------------------------------------------------------------------------

# Import necessary modules
import os
import stat
import gc
from fastapi import APIRouter
from fastapi.responses import JSONResponse

# Create a FastAPI router instance
router = APIRouter()


@router.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

# Health check endpoint for verifying filesystem access and basic I/O
@router.get("/health-check-fs")
async def health_check_fs():
    results = {}  # Dictionary to store health check results

    # Define the directories to check
    temp_dir = "temp_uploads"  # Main temporary storage directory
    output_dir = os.path.join("temp_uploads", "pose_feature_data", "output_video")  # Subdirectory for generated videos

    # Iterate through each directory to perform health checks
    for dir_path in [temp_dir, output_dir]:
        results[dir_path] = {}

        try:
            # Check if the directory exists
            exists = os.path.exists(dir_path)
            results[dir_path]["exists"] = exists

            # If it exists, check its permission mode
            perms = None
            if exists:
                perms = stat.filemode(os.stat(dir_path).st_mode)
            results[dir_path]["permissions"] = perms

            # Construct path for a temporary test file
            test_file = os.path.join(dir_path, "health_check_test.txt")

            # Test write access by trying to create and write to the test file
            try:
                with open(test_file, "w") as f:
                    f.write("health check")
                results[dir_path]["write"] = True
            except Exception as e:
                results[dir_path]["write"] = f"ERROR: {e}"

            # Test read access by reading the contents of the test file
            try:
                with open(test_file, "r") as f:
                    content = f.read()
                results[dir_path]["read"] = content
            except Exception as e:
                results[dir_path]["read"] = f"ERROR: {e}"

            # Test delete access by trying to remove the test file
            try:
                os.remove(test_file)
                results[dir_path]["delete"] = True
            except Exception as e:
                results[dir_path]["delete"] = f"ERROR: {e}"

        except Exception as e:
            # Catch-all for unexpected errors related to the directory
            results[dir_path]["error"] = str(e)

    # Attempt to force garbage collection to ensure memory cleanup works
    try:
        gc.collect()
        results["gc"] = "collected"
    except Exception as e:
        results["gc"] = f"ERROR: {e}"

    # Print the result to the server logs for visibility
    print("HEALTH CHECK FS RESULTS:", results)

    # Return the results as a JSON response
    return JSONResponse(results)


