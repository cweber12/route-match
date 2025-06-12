import os
import cv2
import numpy as np
from .draw_points import draw_pose

def write_pose_video(output_path, ref_img, transformed, fps=24):
    if not transformed:
        print("No frames to write.")
        return

    frame_keys = sorted(transformed.keys())
    height, width = ref_img.shape[:2]
    
    if ref_img.dtype != 'uint8' or (ref_img.ndim == 2 or ref_img.shape[2] != 3):
        raise ValueError("ref_img must be a color image (H, W, 3) with dtype uint8")
    
    # Ensure even width/height for H.264 compatibility
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    print(f"Writing video to {output_path}")

    for frame_num in frame_keys:
        print(f"Writing frame {frame_num}")
        frame = ref_img.copy()
        draw_pose(frame, transformed[frame_num])
        writer.write(frame)

    writer.release()
    print("Video writing complete.")
