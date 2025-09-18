
import threading
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
import logging

JOBS = {}
_LOCK = threading.Lock()
_EXEC = ThreadPoolExecutor(max_workers=2)
logger = logging.getLogger("job_manager")

def _now():
    return int(time.time())

def submit_job(fn: Callable, *args, **kwargs) -> str:
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": _now(),
        "finished_at": None,
        "meta": {},
    }
    with _LOCK:
        JOBS[job_id] = job
    logger.info(f"Job {job_id} submitted. Pending execution.")
    def _runner():
        logger.info(f"Job {job_id} started.")
        with _LOCK:
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["started_at"] = _now()
            JOBS[job_id]["progress"] = 0
        try:
            res = fn(job_id, *args, **kwargs)
            with _LOCK:
                JOBS[job_id]["status"] = "success"
                JOBS[job_id]["result"] = res
                JOBS[job_id]["finished_at"] = _now()
                JOBS[job_id]["progress"] = 100
            logger.info(f"Job {job_id} finished successfully.")
        except Exception as e:
            with _LOCK:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = str(e)
                JOBS[job_id]["finished_at"] = _now()
            logger.error(f"Job {job_id} failed: {e}")
    _EXEC.submit(_runner)
    return job_id

def update_job_progress(job_id: str, percent: float):
    with _LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        try:
            p = float(percent)
        except Exception:
            return
        if p < 0:
            p = 0
        if p > 100:
            p = 100
        job["progress"] = p
        # if progress reached 100, ensure status/finished set
        if p >= 100 and job.get("status") == "running":
            job["status"] = "success"
            job["finished_at"] = _now()

def get_job(job_id: str) -> dict | None:
    with _LOCK:
        return JOBS.get(job_id)
