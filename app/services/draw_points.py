import cv2
import numpy as np

# Define the pose connections (from MediaPipe)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (23, 24), (11, 12), (12, 24), (11, 23)
]

# Define left and right landmark indices (from MediaPipe)
LEFT_LANDMARKS = set([
    11, 13, 15, 17, 19, 21,
    23, 25, 27, 29, 31
])

RIGHT_LANDMARKS = set([
    12, 14, 16, 18, 20, 22,
    24, 26, 28, 30, 32
])

# Function to compute the affine transformation matrix using matched 
# SIFT keypoints. 
def apply_transform(T, landmarks):
    transformed = {}
    for lm in landmarks:
        pt = np.array([[lm["x"], lm["y"]]], dtype=np.float32)
        new_pt = cv2.transform(pt[None, :, :], T)[0][0]
        transformed[lm["landmark_number"]] = new_pt
    return transformed

# use the transformation matrix to draw the pose on the image at the 
# correct coordinates
def draw_pose(img, landmarks):
    h, w = img.shape[:2]
    line_thickness = max(2, int(round(min(w, h) * 0.002)))
    circle_radius = max(2, int(round(min(w, h) * 0.004)))

    for start, end in POSE_CONNECTIONS:
        if start in landmarks and end in landmarks:
            pt1 = landmarks[start]
            pt2 = landmarks[end]
            # Ensure both are iterable (x, y)
            if not (isinstance(pt1, (list, tuple, np.ndarray)) and len(pt1) == 2):
                continue
            if not (isinstance(pt2, (list, tuple, np.ndarray)) and len(pt2) == 2):
                continue
            pt1 = tuple(np.round(pt1).astype(int))
            pt2 = tuple(np.round(pt2).astype(int))

            # Use a bright lime green for all lines
            color = (100, 255, 0)  # BGR for bright lime green
            # Draw anti-aliased, thicker lines for sharpness
            cv2.line(img, pt1, pt2, color, max(line_thickness, 4), lineType=cv2.LINE_AA)

    for idx, pt in landmarks.items():
        # Ensure pt is (x, y)
        if not (isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 2):
            continue
        pt = tuple(np.round(pt).astype(int))

        # Use a bright blue for all points, and make them smaller
        color = (0, 100, 255)  # BGR for bright blue
        cv2.circle(img, pt, max(circle_radius // 2, 2), color, -1, lineType=cv2.LINE_AA)

