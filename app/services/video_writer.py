import os
import cv2
import numpy as np
from .draw_points import draw_pose

def smooth_alpha(alpha):
    # Cosine interpolation for smoother motion
    return (1 - np.cos(np.pi * alpha)) / 2

def write_pose_video(output_path, ref_img, transformed, fps=24):
    if not transformed:
        print("No frames to write.")
        return
    frame_keys = sorted(transformed.keys())
    height, width = ref_img.shape[:2]
    if ref_img.dtype != 'uint8' or (ref_img.ndim == 2 or ref_img.shape[2] != 3):
        raise ValueError("ref_img must be a color image (H, W, 3) with dtype uint8")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    print(f"Writing video to {output_path}")

    for i in range(len(frame_keys) - 1):
        f1, f2 = frame_keys[i], frame_keys[i + 1]
        print(f"Processing frames {f1} to {f2}")
        print(f"Keys in f1: {list(transformed[f1].keys())}, Keys in f2: {list(transformed[f2].keys())}")

        # Draw frame f1
        frame_start = ref_img.copy()
        draw_pose(frame_start, transformed[f1])
        print(f"Writing frame with shape {frame_start.shape}, dtype {frame_start.dtype}")
        writer.write(frame_start)

        gap = f2 - f1
        print(f"Gap between frames: {gap}")
        if gap <= 1:
            continue

        # Interpolate in between
        for j in range(1, gap):
            raw_alpha = j / gap
            alpha = smooth_alpha(raw_alpha)
            print(f"Interpolating frame {f1} to {f2}, step {j}/{gap}, alpha={alpha:.3f}")
            interp = {
                k: (1 - alpha) * transformed[f1][k] + alpha * transformed[f2][k]
                for k in transformed[f1] if k in transformed[f2]
            }
            frame_interp = ref_img.copy()
            draw_pose(frame_interp, interp)
            interp_frame_num = f1 + j
            print(f"Writing interpolated frame {interp_frame_num}")
            writer.write(frame_interp)

    # Draw last frame
    frame_end = ref_img.copy()
    draw_pose(frame_end, transformed[frame_keys[-1]])
    writer.write(frame_end)
    writer.release()
