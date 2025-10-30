# app/api/routers/download.py
# ----------------------------------------------------------------------------------------------------------
# This module provides endpoints to download executable files from S3 and serve them to the frontend.
# It includes functions to stream files from S3, get file metadata, and handle download requests securely.
# ----------------------------------------------------------------------------------------------------------

import os
import io
import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# S3 configuration for executable downloads
EXECUTABLE_BUCKET = "process-executable"
EXECUTABLE_KEY = "RouteScanner.exe"

def get_s3_client():
    """Get configured S3 client using environment variables"""
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

@router.get("/download-executable")
async def download_executable():
    """
    Download RouteScanner.exe from S3 and stream it to the frontend
    """
    try:
        s3_client = get_s3_client()
        
        # Check if file exists first
        try:
            head_response = s3_client.head_object(Bucket=EXECUTABLE_BUCKET, Key=EXECUTABLE_KEY)
            file_size = head_response.get('ContentLength', 0)
            logger.info(f"Found executable {EXECUTABLE_KEY} in bucket {EXECUTABLE_BUCKET}, size: {file_size}")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"Executable not found: {EXECUTABLE_KEY}")
                raise HTTPException(status_code=404, detail="RouteScanner.exe not found in S3")
            else:
                logger.error(f"S3 access error during head_object: {e}")
                raise HTTPException(status_code=500, detail=f"Error accessing S3: {str(e)}")
        
        # Download and stream the file
        try:
            logger.info(f"Starting download of {EXECUTABLE_KEY}")
            response = s3_client.get_object(Bucket=EXECUTABLE_BUCKET, Key=EXECUTABLE_KEY)
            
            # Read the entire file content to ensure complete transfer
            file_content = response['Body'].read()
            logger.info(f"Successfully read {len(file_content)} bytes from S3")
            
            # Verify we got the expected amount of data
            if len(file_content) != file_size:
                logger.error(f"File size mismatch: expected {file_size}, got {len(file_content)}")
                raise HTTPException(status_code=500, detail="File download incomplete")
            
            logger.info(f"Successfully prepared streaming response for {file_size} bytes")
            
            return StreamingResponse(
                io.BytesIO(file_content),
                media_type="application/x-msdownload",
                headers={
                    "Content-Disposition": "attachment; filename=RouteScanner.exe",
                    "Content-Length": str(file_size),
                    "Content-Type": "application/x-msdownload",
                    "Access-Control-Expose-Headers": "Content-Disposition, Content-Length",
                    "Accept-Ranges": "bytes", 
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Expose-Headers": "Content-Disposition, Content-Length, Content-Type",
                }
            )
            
        except ClientError as e:
            logger.error(f"Error downloading from S3: {e}")
            raise HTTPException(status_code=500, detail="Failed to download executable from S3")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in download_executable: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during download")

@router.get("/executable-info")
async def get_executable_info():
    """
    Get metadata about the RouteScanner.exe file without downloading it
    
    Returns:
        JSONResponse: File metadata including size, last modified, and download URL
        
    Raises:
        HTTPException: If file not found or S3 access fails
    """
    try:
        s3_client = get_s3_client()
        
        try:
            response = s3_client.head_object(Bucket=EXECUTABLE_BUCKET, Key=EXECUTABLE_KEY)
            
            file_size = response.get('ContentLength', 0)
            last_modified = response.get('LastModified')
            
            logger.info(f"Retrieved info for {EXECUTABLE_KEY}: {file_size} bytes")
            
            return JSONResponse({
                "filename": "RouteScanner.exe",
                "size": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "last_modified": last_modified.isoformat() if last_modified else None,
                "download_url": "/api/download-executable",
                "bucket": EXECUTABLE_BUCKET,
                "key": EXECUTABLE_KEY
            })
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"Executable not found during info request: {EXECUTABLE_KEY}")
                raise HTTPException(status_code=404, detail="RouteScanner.exe not found in S3")
            else:
                logger.error(f"S3 access error during head_object: {e}")
                raise HTTPException(status_code=500, detail=f"Error accessing S3: {str(e)}")
                
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_executable_info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error getting file info")

@router.get("/download-status")
async def get_download_status():
    """
    Check if the executable download service is available
    
    Returns:
        JSONResponse: Service status and configuration info
    """
    try:
        s3_client = get_s3_client()
        
        # Test S3 connectivity
        try:
            s3_client.head_bucket(Bucket=EXECUTABLE_BUCKET)
            s3_available = True
            s3_error = None
        except Exception as e:
            s3_available = False
            s3_error = str(e)
            logger.warning(f"S3 bucket not accessible: {e}")
        
        return JSONResponse({
            "service": "executable-download",
            "status": "available" if s3_available else "unavailable",
            "s3_bucket": EXECUTABLE_BUCKET,
            "s3_key": EXECUTABLE_KEY,
            "s3_accessible": s3_available,
            "s3_error": s3_error,
            "endpoints": {
                "download": "/api/download-executable",
                "info": "/api/executable-info",
                "status": "/api/download-status"
            }
        })
        
    except Exception as e:
        logger.error(f"Error checking download status: {e}")
        return JSONResponse({
            "service": "executable-download",
            "status": "error",
            "error": str(e)
        }, status_code=500)