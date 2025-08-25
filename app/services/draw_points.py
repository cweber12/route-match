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
def apply_transform(T, landmarks, force_apply: bool = False):
    if T is None:
        print("Warning: Transform matrix is None, returning original landmarks")
        return landmarks

    # Log transformation matrix details for debugging
    try:
        scale_x = np.sqrt(T[0, 0]**2 + T[0, 1]**2)
        scale_y = np.sqrt(T[1, 0]**2 + T[1, 1]**2)
        rotation_angle = np.arctan2(T[1, 0], T[0, 0])
        rotation_degrees = np.degrees(rotation_angle)
        print(f"Applying transform: scale=({scale_x:.3f}, {scale_y:.3f}), rotation={rotation_degrees:.1f}°")
    except Exception:
        print("Warning: invalid transform format; returning original landmarks")
        return landmarks

    # More lenient validation - only reject truly extreme cases
    def validate_transformation(transform):
        if transform is None:
            return False, None
        try:
            scale_x = np.sqrt(transform[0, 0]**2 + transform[0, 1]**2)
            scale_y = np.sqrt(transform[1, 0]**2 + transform[1, 1]**2)
            rotation_angle = np.arctan2(transform[1, 0], transform[0, 0])
            rotation_degrees = np.degrees(rotation_angle)
        except Exception:
            return False, None

        if scale_x < 0.05 or scale_x > 20.0 or scale_y < 0.05 or scale_y > 20.0:
            return False, (scale_x, scale_y, rotation_degrees)
        if abs(rotation_degrees) > 45:
            return False, (scale_x, scale_y, rotation_degrees)

        return True, (scale_x, scale_y, rotation_degrees)

    valid, metrics = validate_transformation(T)
    if not valid:
        if not force_apply:
            print("Transformation matrix validation failed, returning original landmarks")
            return landmarks
        else:
            # force-apply: clamp linear components to avoid extreme warps
            print(f"Transformation validation failed, force_applying with clamping; metrics={metrics}")
            # clamp column norms
            A = np.asarray(T, dtype=np.float64)[:, :2].copy()
            col0 = np.linalg.norm(A[:, 0])
            col1 = np.linalg.norm(A[:, 1])
            max_scale = 12.0
            min_scale = 1.0 / max_scale
            sf0 = 1.0
            sf1 = 1.0
            if col0 > 0:
                if col0 > max_scale:
                    sf0 = max_scale / col0
                elif col0 < min_scale:
                    sf0 = min_scale / col0
            if col1 > 0:
                if col1 > max_scale:
                    sf1 = max_scale / col1
                elif col1 < min_scale:
                    sf1 = min_scale / col1
            A[:, 0] = A[:, 0] * sf0
            A[:, 1] = A[:, 1] * sf1
            T = np.hstack([A, np.asarray(T, dtype=np.float64)[:, 2:3]])

    transformed = {}
    transform_count = 0

    # Helper to apply affine to (x,y)
    def apply_to_point(x, y):
        new_x = T[0, 0] * float(x) + T[0, 1] * float(y) + T[0, 2]
        new_y = T[1, 0] * float(x) + T[1, 1] * float(y) + T[1, 2]
        return float(new_x), float(new_y)

    # Support dict keyed by index: {idx: (x,y) | {'x':..,'y':..} }
    if isinstance(landmarks, dict):
        for k, v in landmarks.items():
            try:
                idx = int(k)
            except Exception:
                # skip non-integer keys
                continue
            if v is None:
                continue
            # v can be tuple/list (x,y) or dict with x,y
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                x, y = v[0], v[1]
            elif isinstance(v, dict) and "x" in v and "y" in v:
                x, y = v["x"], v["y"]
            else:
                # unknown format, skip
                continue
            try:
                new_x, new_y = apply_to_point(x, y)
                transformed[int(idx)] = (new_x, new_y)
                transform_count += 1
                if transform_count <= 3:
                    print(f"Landmark {idx}: ({x:.1f}, {y:.1f}) -> ({new_x:.1f}, {new_y:.1f})")
            except Exception as e:
                print(f"Error transforming landmark {idx}: {e}")
                continue

    # Support list/tuple of landmark dicts (legacy format)
    elif isinstance(landmarks, (list, tuple)):
        for lm in landmarks:
            if not isinstance(lm, dict):
                continue
            idx = lm.get("landmark_number") or lm.get("index") or lm.get("id")
            if idx is None:
                continue
            if "x" not in lm or "y" not in lm:
                continue
            try:
                x, y = float(lm["x"]), float(lm["y"])
                new_x, new_y = apply_to_point(x, y)
                transformed[int(idx)] = (new_x, new_y)
                transform_count += 1
                if transform_count <= 3:
                    print(f"Landmark {idx}: ({x:.1f}, {y:.1f}) -> ({new_x:.1f}, {new_y:.1f})")
            except Exception as e:
                print(f"Error transforming landmark {lm}: {e}")
                continue
    else:
        # Unknown landmark container type - return as is but warn
        print(f"Warning: unsupported landmarks container type: {type(landmarks)}")
        return landmarks

    print(f"Successfully transformed {transform_count} landmarks")
    return transformed

# use the transformation matrix to draw the pose on the image at the 
# correct coordinates
def draw_pose(img, landmarks, line_color=(100, 255, 0), point_color=(0, 100, 255)):
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

            # Draw anti-aliased, thicker lines for sharpness
            cv2.line(img, pt1, pt2, line_color, max(line_thickness, 4), lineType=cv2.LINE_AA)

    for idx, pt in landmarks.items():
        # Ensure pt is (x, y)
        if not (isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 2):
            continue
        pt = tuple(np.round(pt).astype(int))

        # Make them smaller
        cv2.circle(img, pt, max(circle_radius // 2, 2), point_color, -1, lineType=cv2.LINE_AA)

def rgb_to_bgr(color):
    if isinstance(color, (list, tuple)) and len(color) == 3:
        return (color[2], color[1], color[0])
    return color

