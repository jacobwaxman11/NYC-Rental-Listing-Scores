"""Pipeline job runner — scrape / backfill / score, driven from the web UI.

At most one job runs at a time. We spawn the existing CLI scripts as
subprocesses (the same Python interpreter that's serving the app, so the venv
is inherited), capture stdout line by line into a shared buffer, and let the
browser poll ``/api/run/log`` for incremental progress. Params are passed as
discrete argv elements (never through a shell), so UI input can't inject
commands. ``start_job`` accepts a list of (stage, params) so "Run all" can
chain the whole pipeline in one background job.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading

# Repo root is the parent of this package, so subprocess paths (e.g.
# ``scrape_listings.py``) resolve regardless of where the server is launched.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_JOB: dict = {
    "stage": None,
    "status": "idle",     # idle | running | done | failed | stopped
    "lines": [],          # captured stdout/stderr lines
    "returncode": None,
    "proc": None,
}
_JOB_LOCK = threading.Lock()

_STAGES = {
    "scrape": "scrape_listings.py",
    "backfill": "backfill_details.py",
    "score": "score_listings.py",
}

# The order stages run in when "Run all" chains the whole pipeline.
PIPELINE_ORDER = ["scrape", "backfill", "score"]


def _build_argv(stage: str, params: dict) -> list[str]:
    """Translate posted params into a safe argv list for the stage's script.

    Only known flags are forwarded, and each value is cast (ints/floats) or
    passed verbatim as its own argv element — no shell interpolation.
    """
    argv = [_STAGES[stage]]

    def add(flag: str, key: str, cast=str) -> None:
        v = params.get(key)
        if v is None or v == "":
            return
        try:
            argv.extend([flag, str(cast(v))])
        except (TypeError, ValueError):
            pass  # ignore an unparseable value rather than crash the launch

    if stage == "scrape":
        add("--areas", "areas")
        add("--price-min", "price_min", int)
        add("--price-max", "price_max", int)
        add("--max-listings", "max_listings", int)
        add("--max-pages", "max_pages", int)
    elif stage == "backfill":
        add("--impersonate", "impersonate")
        add("--min-delay", "min_delay", float)
        add("--max-delay", "max_delay", float)
        add("--max-listings", "max_listings", int)
    elif stage == "score":
        add("--provider", "provider")
        add("--model", "model")
        add("--max-listings", "max_listings", int)
    return argv


def _spawn(stage: str, params: dict) -> "subprocess.Popen":
    """Start one stage's subprocess and record it as the current job's proc."""
    argv = _build_argv(stage, params)
    proc = subprocess.Popen(
        [sys.executable, "-u", *argv],
        cwd=_REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with _JOB_LOCK:
        _JOB["stage"] = stage
        _JOB["proc"] = proc
        _JOB["lines"].append("$ python " + " ".join(argv))
    return proc


def _stream(proc: "subprocess.Popen") -> int:
    """Stream a proc's output into the shared buffer; return its exit code."""
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        with _JOB_LOCK:
            _JOB["lines"].append(line.rstrip("\n"))
    proc.stdout.close()
    return proc.wait()


def _worker(items: list) -> None:
    """Run a sequence of (stage, params) in order. Each stage streams live; the
    chain stops on a non-zero exit or if the user hits Stop. A single-stage list
    behaves exactly like running that one stage."""
    seq = len(items) > 1
    overall_rc = 0
    for i, (stage, params) in enumerate(items):
        with _JOB_LOCK:
            if _JOB["status"] != "running":      # user stopped before this stage
                break
            if seq:
                _JOB["lines"].append(f"\n══ [{i + 1}/{len(items)}] {stage} ══")
        rc = _stream(_spawn(stage, params))
        with _JOB_LOCK:
            stopped = _JOB["status"] == "stopped"
            _JOB["lines"].append(f"── {stage} exited with code {rc} ──")
        if stopped:
            overall_rc = rc or 1
            break
        if rc != 0:
            overall_rc = rc
            if seq:
                with _JOB_LOCK:
                    _JOB["lines"].append(f"✗ {stage} failed (exit {rc}) — stopping the pipeline.")
            break
    with _JOB_LOCK:
        _JOB["returncode"] = overall_rc
        if _JOB["status"] == "running":
            _JOB["status"] = "done" if overall_rc == 0 else "failed"
        if seq:
            _JOB["lines"].append("══ pipeline " + ("done ✓" if overall_rc == 0 else "stopped") + " ══")


def start_job(items: list) -> tuple[bool, str]:
    """Start a list of (stage, params) tuples as one job. Returns (started, msg)."""
    with _JOB_LOCK:
        if _JOB["status"] == "running":
            return False, "A job is already running — wait for it to finish or stop it."
        _JOB.update(stage=items[0][0], status="running", lines=[], returncode=None, proc=None)
    threading.Thread(target=_worker, args=(items,), daemon=True).start()
    return True, "started"
