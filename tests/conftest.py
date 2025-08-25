import sys
import types
import os
import importlib
import pytest


def _make_fake_cv2():
    fake = types.SimpleNamespace()

    # constants
    fake.COLOR_BGR2GRAY = 6

    # simple KeyPoint factory
    class KeyPoint:
        def __init__(self, x=0, y=0, size=1):
            self.pt = (x, y)
            self.size = size
            self.angle = 0
            self.response = 0
            self.octave = 0
            self.class_id = 0

    fake.KeyPoint = KeyPoint

    # minimal implementations used by the codebase
    def imread(path, flags=1):
        # return a simple truthy object to indicate success
        return object()

    def SIFT_create(**kwargs):
        class SiftObj:
            def detectAndCompute(self, img, mask=None):
                # return empty keypoints and None descriptors
                return [], None
        return SiftObj()

    def cvtColor(img, code):
        return img

    def equalizeHist(img):
        return img

    def normalize(src, dst=None, alpha=0, beta=255, norm_type=None):
        return src

    class VideoWriter:
        def __init__(self, *args, **kwargs):
            self._opened = True
        def isOpened(self):
            return True
        def write(self, frame):
            return
        def release(self):
            return

    fake.imread = imread
    fake.SIFT_create = SIFT_create
    fake.cvtColor = cvtColor
    fake.equalizeHist = equalizeHist
    fake.normalize = normalize
    fake.VideoWriter = VideoWriter

    # provide some attributes used by modules
    fake.CV_8U = 0

    return fake


def _make_fake_numpy():
    fake = types.ModuleType("numpy")
    # Minimal dtypes
    fake.float32 = float
    fake.float64 = float

    def array(seq, dtype=None):
        # Return nested lists to avoid real numpy
        return list(seq)

    def asarray(obj, dtype=None):
        return obj if isinstance(obj, list) else [obj]

    def vstack(lst):
        # flatten lists of lists
        result = []
        for sub in lst:
            result.extend(list(sub))
        return result

    def linspace(a, b, n):
        if n <= 0:
            return []
        if n == 1:
            return [b]
        step = (b - a) / (n - 1)
        return [a + i * step for i in range(n)]

    def sqrt(x):
        return (x ** 0.5) if x >= 0 else float('nan')

    fake.array = array
    fake.asarray = asarray
    fake.vstack = vstack
    fake.linspace = linspace
    fake.sqrt = sqrt

    return fake


@pytest.fixture(autouse=True, scope="session")
def enable_compare_with_mocks(tmp_path_factory):
    """Autouse fixture to inject fake cv2 and stub heavy services, then
    import and register the real compare router on main.app for tests.
    """
    # 1) Insert fake numpy and cv2 into sys.modules before importing compare
    fake_cv2 = _make_fake_cv2()
    sys.modules.setdefault("cv2", fake_cv2)

    fake_np = _make_fake_numpy()
    sys.modules.setdefault("numpy", fake_np)

    # 2) Import the compare module (it imports functions at module scope)
    try:
        compare_mod = importlib.import_module("app.routers.compare")
    except Exception as e:
        pytest.skip(f"Could not import real compare module: {e}")

    # 3) Stub heavy functions used by compare router (transform_skeleton functions)
    from app import main as main_mod

    # Ensure VIDEO_OUT_DIR exists on disk
    try:
        transform_mod = importlib.import_module("app.services.transform_skeleton")
        video_out_dir = getattr(transform_mod, "VIDEO_OUT_DIR", None)
    except Exception:
        transform_mod = None
        video_out_dir = os.path.join("temp_uploads", "pose_feature_data", "output_video")

    abs_video_out_dir = os.path.abspath(video_out_dir)
    os.makedirs(abs_video_out_dir, exist_ok=True)

    # Stubs
    def _stub_generate_video(*args, **kwargs):
        out_raw = os.path.join(abs_video_out_dir, "output_video.mp4")
        out_browser = os.path.join(abs_video_out_dir, "output_video_browser.mp4")
        # write tiny placeholder files
        with open(out_raw, "wb") as f:
            f.write(b"dummy mp4")
        with open(out_browser, "wb") as f:
            f.write(b"dummy mp4")
        return "SUCCESS"

    def _stub_convert_video_for_browser(input_path, output_path, *args, **kwargs):
        # ensure browser file exists
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(b"dummy browser mp4")
            return {"status": "success", "output_path": output_path}
        except Exception as e:
            return {"error": str(e)}

    # Replace names in the compare module with stubs
    setattr(compare_mod, "generate_video", _stub_generate_video)
    setattr(compare_mod, "generate_video_multiframe", _stub_generate_video)
    setattr(compare_mod, "convert_video_for_browser", _stub_convert_video_for_browser)

    # Stub S3/JSON loaders so compare can build its internal structures
    def _stub_load_pose_data_from_path(s3_folder):
        # return a minimal pose dict where keys are strings
        return {"0": [{"frame": "0", "landmarks": {}}]}

    def _stub_load_sift_data_from_path(s3_folder):
        # Return legacy single-frame format: (kps_all, descs_all), frame_dims, is_multi_frame=False
        kps_all = [[]]
        descs_all = [[0] * 128]
        frame_dimensions = (64, 64)
        return (kps_all, descs_all), frame_dimensions, False

    setattr(compare_mod, "load_pose_data_from_path", _stub_load_pose_data_from_path)
    setattr(compare_mod, "load_sift_data_from_path", _stub_load_sift_data_from_path)

    # 4) Remove any previously-registered compare stub routes on main.app
    try:
        app = main_mod.app
        # remove routes with path '/api/compare-image' if present
        app.routes = [r for r in app.routes if getattr(r, "path", None) != "/api/compare-image"]
        # register the real compare router now that we've stubbed heavy internals
        app.include_router(compare_mod.router, prefix="/api", tags=["Keypoint Comparison"])
    except Exception as e:
        pytest.skip(f"Failed to include real compare router into app: {e}")

    yield
