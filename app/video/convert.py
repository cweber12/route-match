import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_video_for_browser(
    input_path: str, 
    output_path: str, 
    async_mode: bool = False,
    quality_preset: str = "fast"
) -> Optional[Dict[str, Any]]:
    
    # Input validation
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input video file not found: {input_path}")
        return {"error": "Input file not found", "path": input_path}
    
    if input_file.stat().st_size == 0:
        logger.error(f"Input video file is empty: {input_path}")
        return {"error": "Input file is empty", "path": input_path}
    
    # Check if output already exists and is newer
    output_file = Path(output_path)
    if (output_file.exists() and 
        output_file.stat().st_mtime > input_file.stat().st_mtime):
        logger.info(f"Output file already up-to-date: {output_path}")
        return {"status": "already_exists", "path": output_path}
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check ffmpeg availability
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        logger.error("ffmpeg executable not found in PATH")
        return {"error": "ffmpeg not found"}
    
    def _do_conversion():
        """Internal conversion function"""
        # Quality presets for different use cases
        presets = {
            "fast": {
                "crf": "28",
                "preset": "ultrafast",
                "profile": "baseline",
                "level": "3.0"
            },
            "balanced": {
                "crf": "23",
                "preset": "fast", 
                "profile": "main",
                "level": "3.1"
            },
            "quality": {
                "crf": "20",
                "preset": "medium",
                "profile": "high",
                "level": "4.0"
            }
        }
        
        settings = presets.get(quality_preset, presets["balanced"])
        
        # Optimized ffmpeg command
        cmd = [
            ffmpeg_exe,
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-c:v", "libx264",
            "-crf", settings["crf"],
            "-preset", settings["preset"],
            "-profile:v", settings["profile"],
            "-level", settings["level"],
            "-movflags", "+faststart",  # Enable progressive download
            "-pix_fmt", "yuv420p",  # Ensure compatibility
            "-f", "mp4",  # Force MP4 format
            str(output_path)
        ]
        
        try:
            # Run with timeout and capture output
            result = subprocess.run(
                cmd, 
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Verify output file was created successfully
            if not output_file.exists() or output_file.stat().st_size == 0:
                raise Exception("Output file was not created or is empty")
            
            logger.info(f"Successfully converted video: {output_path}")
            return {
                "status": "success", 
                "input_path": input_path,
                "output_path": output_path,
                "file_size": output_file.stat().st_size
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg conversion timed out for {input_path}")
            return {"error": "conversion_timeout", "path": input_path}
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr}")
            return {"error": "conversion_failed", "details": e.stderr}
        except Exception as e:
            logger.error(f"Unexpected error during conversion: {e}")
            return {"error": "unexpected_error", "details": str(e)}
    
    if async_mode:
        # Run conversion in background thread
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_conversion)
        return {"status": "started", "future": future}
    else:
        # Run synchronously
        return _do_conversion()