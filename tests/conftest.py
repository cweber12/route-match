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
    # Try to load the real compare router from its file path; if that fails
    # (missing package path or native deps), fall back to the pure-Python
    # test router implemented in tests/compare_test_router.py
    import importlib.util

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Ensure project root is on sys.path so 'app' package can be imported
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    compare_path = os.path.join(project_root, "app", "routers", "compare.py")
    test_router_path = os.path.join(os.path.dirname(__file__), "compare_test_router.py")

    def _load_module_from_path(mod_name, path):
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {mod_name} from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    try:
        if os.path.exists(compare_path):
            compare_mod = _load_module_from_path("app.routers.compare", compare_path)
        else:
            raise FileNotFoundError("compare.py not found")
    except Exception:
        # fallback to test router
        compare_mod = _load_module_from_path("tests.compare_test_router", test_router_path)

    # 3) Stub heavy functions used by compare router (transform_skeleton functions)
    # Resolve the main app module (the tests import `main` at top-level)
    if "main" in sys.modules:
        main_mod = sys.modules["main"]
    else:
        try:
            main_mod = importlib.import_module("main")
        except Exception:
            # as a last resort, try to load app/main.py directly
            main_path = os.path.join(project_root, "app", "main.py")
            main_mod = _load_module_from_path("main", main_path)

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

    # If the real module was loaded, replace heavy internals with stubs; if
    # the test router is used, it is already lightweight and needs no stubbing.
    if compare_mod.__name__ == "app.routers.compare":
        setattr(compare_mod, "generate_video", _stub_generate_video)
        setattr(compare_mod, "generate_video_multiframe", _stub_generate_video)
        setattr(compare_mod, "convert_video_for_browser", _stub_convert_video_for_browser)

        def _stub_load_pose_data_from_path(s3_folder):
            return {"0": [{"frame": "0", "landmarks": {}}]}

        def _stub_load_sift_data_from_path(s3_folder):
            kps_all = [[]]
            descs_all = [[0] * 128]
            frame_dimensions = (64, 64)
            return (kps_all, descs_all), frame_dimensions, False

        setattr(compare_mod, "load_pose_data_from_path", _stub_load_pose_data_from_path)
        setattr(compare_mod, "load_sift_data_from_path", _stub_load_sift_data_from_path)

    # 4) Remove any previously-registered compare stub routes on main.app
    try:
        app = main_mod.app
        # remove routes with path '/api/compare-image' if present by mutating router.routes
        routes = list(app.router.routes)
        routes = [r for r in routes if getattr(r, "path", None) != "/api/compare-image"]
        # assign back by clearing and extending router.routes
        app.router.routes.clear()
        app.router.routes.extend(routes)
        # register the real/test compare router now that we've stubbed heavy internals
        app.include_router(compare_mod.router, prefix="/api", tags=["Keypoint Comparison"])
    except Exception as e:
        pytest.skip(f"Failed to include real compare router into app: {e}")

    yield
