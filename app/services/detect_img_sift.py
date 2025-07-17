import cv2
import numpy as np

def detect_sift(image, sift_config=None, use_clahe=False, clahe_config=None, use_hist_eq=True, use_normalize=True, bbox=None, detector=None):
    sift_config = sift_config or {}
    
    # Force garbage collection before creating SIFT detector
    import gc
    gc.collect()
    
    # Always create a fresh SIFT detector - remove reuse logic to prevent state contamination
    if detector is not None:
        print("⚠ Reusing detector is discouraged. Creating fresh one.")
    
    # Create a fresh SIFT detector for each call to avoid any state contamination
    sift = cv2.SIFT_create(**sift_config)
    print("✔ Created fresh SIFT detector")
    
    # Log SIFT configuration
    if sift_config:
        print(f"SIFT configuration: {sift_config}")

    # Crop to bbox if provided
    offset_x, offset_y = 0, 0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        offset_x, offset_y = x1, y1
        image = image[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization if requested
    if use_hist_eq:
        gray = cv2.equalizeHist(gray)

    if use_clahe:
        clahe_config = clahe_config or {}
        clip_limit = clahe_config.get("clipLimit", 2.0)
        tile_grid_size = clahe_config.get("tileGridSize", (8, 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        gray = clahe.apply(gray)

    # Apply normalization if requested (scales pixel values to full 0-255 range)
    if use_normalize:
        gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Run SIFT
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Always clean up detector since we always create a fresh one
    del sift
    gc.collect()
    
    # Validate SIFT results immediately
    if keypoints is None or descriptors is None:
        print("❌ SIFT detection failed - got None keypoints or descriptors")
        return [], None
    
    if len(keypoints) == 0 or descriptors.shape[0] == 0:
        print("❌ SIFT detection failed - got empty keypoints or descriptors")
        return [], None
    
    print(f"✔ SIFT detection successful: {len(keypoints)} keypoints, descriptor shape: {descriptors.shape}")
    
    # Log preprocessing steps
    print(f"Preprocessing steps: use_hist_eq={use_hist_eq}, use_clahe={use_clahe}, use_normalize={use_normalize}")
    if use_clahe and clahe_config:
        print(f"CLAHE configuration: clipLimit={clahe_config.get('clipLimit', 2.0)}, tileGridSize={clahe_config.get('tileGridSize', (8, 8))}")

    # Log bounding box if provided
    if bbox:
        print(f"Cropping image to bbox: {bbox}")

    # Adjust keypoint coordinates back to full image coordinate system if bbox was used
    if bbox is not None and keypoints is not None:
        adjusted_keypoints = []
        for kp in keypoints:
            new_kp = cv2.KeyPoint(
                x=kp.pt[0] + offset_x,
                y=kp.pt[1] + offset_y,
                size=kp.size,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id
            )
            adjusted_keypoints.append(new_kp)
        keypoints = adjusted_keypoints
    
    return keypoints, descriptors

