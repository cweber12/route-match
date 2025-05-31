import cv2
import numpy as np

def detect_sift(image, sift_config=None, use_clahe=False, clahe_config=None, use_hist_eq=True, use_normalize=True, bbox=None):
    sift_config = sift_config or {}
    sift = cv2.SIFT_create(**sift_config)

    # Crop to bbox if provided
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        image = image[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization if requested
    if use_hist_eq:
        gray = cv2.equalizeHist(gray)

    # Apply normalization if requested (scales pixel values to full 0-255 range)
    if use_normalize:
        gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Apply CLAHE if requested (should not combine with hist_eq/normalize)
    #if use_clahe:
        #clahe_config = clahe_config or {}
        #clip_limit = clahe_config.get("clipLimit", 2.0)
        #tile_grid_size = clahe_config.get("tileGridSize", (8, 8))
        #clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        #gray = clahe.apply(gray)

    # Run SIFT
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

