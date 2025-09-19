# tools/export_routes.py
from __future__ import annotations
import os, sys, json, importlib, traceback, logging, warnings
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import List, Dict

# Silence noisy libs & warnings
os.environ.setdefault("ROUTE_EXPORT", "1")         # you can check this in your app to skip heavy init
for name in ("botocore", "boto3", "numpy", "PIL", "uvicorn"):
    logging.getLogger(name).setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

try:
    from fastapi import FastAPI
    from fastapi.routing import APIRoute
except Exception as e:
    print(f"[export_routes] FastAPI import failed: {e}", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr); raise SystemExit(code)

def load_app(app_spec: str) -> FastAPI:
    if ":" not in app_spec:
        app_spec += ":app"
    mod_name, var_name = app_spec.split(":", 1)
    print(f"[export_routes] Importing {mod_name}:{var_name}", file=sys.stderr)

    # swallow prints during import
    buff_out, buff_err = StringIO(), StringIO()
    with redirect_stdout(buff_out), redirect_stderr(buff_err):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            traceback.print_exc()
            die(f"[export_routes] FAILED importing module '{mod_name}'")
        try:
            app_obj = getattr(mod, var_name)
        except Exception:
            traceback.print_exc()
            die(f"[export_routes] Module '{mod_name}' has no attr '{var_name}'")

    if not isinstance(app_obj, FastAPI):
        die(f"[export_routes] {mod_name}:{var_name} is not a FastAPI instance (got {type(app_obj)})")
    return app_obj

def safe_name(obj) -> str | None:
    try:
        return getattr(obj, "__name__", None)
    except Exception:
        return None

def route_info(r: APIRoute) -> Dict:
    # Everything behind try/except so one quirky route doesn’t kill the run
    try:
        req_name = None
        body_field = getattr(r, "body_field", None)
        if body_field is not None:
            # some Starlette versions use .type_ attr
            t = getattr(body_field, "type_", None)
            req_name = safe_name(t) if t else None

        resp_name = None
        resp_model = getattr(r, "response_model", None)
        if resp_model is not None:
            resp_name = safe_name(resp_model)

        deps = []
        dependant = getattr(r, "dependant", None)
        if dependant and getattr(dependant, "dependencies", None):
            for d in dependant.dependencies:
                deps.append(safe_name(getattr(d, "call", None)) or str(getattr(d, "call", "")))

        endpoint = r.endpoint
        endpoint_fq = f"{getattr(endpoint, '__module__', '')}.{getattr(endpoint, '__name__', '')}"
        endpoint_file = getattr(getattr(endpoint, "__code__", None), "co_filename", None)
        endpoint_line = getattr(getattr(endpoint, "__code__", None), "co_firstlineno", None)

        return {
            "path": r.path,
            "methods": sorted(list(r.methods or [])),
            "endpoint": endpoint_fq,
            "dependencies": deps,
            "request_model": req_name,
            "response_model": resp_name,
            "name": getattr(r, "name", None),
            "summary": getattr(r, "summary", None),
            "tags": getattr(r, "tags", []) or [],
            "endpoint_file": endpoint_file,
            "endpoint_line": endpoint_line,
        }
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return {
            "path": getattr(r, "path", "<unknown>"),
            "methods": sorted(list(getattr(r, "methods", []) or [])),
            "endpoint": "<error>",
            "dependencies": [],
            "request_model": None,
            "response_model": None,
            "name": getattr(r, "name", None),
            "summary": getattr(r, "summary", None),
            "tags": getattr(r, "tags", []) or [],
            "error": "route_info_failed",
        }

def main(argv: List[str]) -> int:
    out_path: Path | None = None
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            out_path = Path(argv[i + 1])

    app_spec = os.getenv("APP_MODULE")
    if not app_spec:
        die("[export_routes] APP_MODULE is not set (e.g., 'app.main:app'). Set it and retry.", 3)

    app = load_app(app_spec)

    # build routes with stdout/stderr silenced as well (in case handlers print on inspection)
    buff_out, buff_err = StringIO(), StringIO()
    with redirect_stdout(buff_out), redirect_stderr(buff_err):
        routes = [route_info(r) for r in app.routes if isinstance(r, APIRoute)]

    payload = json.dumps(routes, indent=2)
    out_path = out_path or (ROOT / "routes.json")
    out_path.write_text(payload, encoding="utf-8")
    print(f"[export_routes] Wrote {len(routes)} routes to {out_path}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))



