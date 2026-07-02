"""Pipeline control routes: the /run control panel plus the endpoints that
launch, poll, and stop the scrape → backfill → score job.

The actual job runner (subprocess management, log buffering) lives in
:mod:`webapp.pipeline`; these routes are a thin HTTP layer over it.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from webapp.areas import AREA_OPTIONS
from webapp.pipeline import (
    _JOB, _JOB_LOCK, _STAGES, PIPELINE_ORDER, start_job,
)
from webapp.state import _STATE

bp = Blueprint("pipeline", __name__)


@bp.route("/run")
def run_panel():
    """Control-panel page: set params, launch a stage, watch live progress."""
    return render_template(
        "run.html", ai=_STATE["ai"], db_path=_STATE["db_path"], areas=AREA_OPTIONS
    )


@bp.route("/api/run/all", methods=["POST"])
def api_run_all():
    """Run scrape → backfill → score back-to-back as one server-side job, so it
    keeps going after you close the tab. Body: {"scrape": {...}, "backfill":
    {...}, "score": {...}} (any stage's params may be omitted)."""
    body = request.get_json(silent=True) or {}
    items = [(s, body.get(s) or {}) for s in PIPELINE_ORDER]
    started, msg = start_job(items)
    if not started:
        return jsonify({"ok": False, "error": msg}), 409
    return jsonify({"ok": True, "stage": "all"})


@bp.route("/api/run/<stage>", methods=["POST"])
def api_run(stage: str):
    if stage not in _STAGES:
        return jsonify({"ok": False, "error": f"unknown stage {stage!r}"}), 400
    params = request.get_json(silent=True) or {}
    started, msg = start_job([(stage, params)])
    if not started:
        return jsonify({"ok": False, "error": msg}), 409
    return jsonify({"ok": True, "stage": stage})


@bp.route("/api/run/log")
def api_run_log():
    """Return job status plus any log lines after ``cursor`` (incremental poll)."""
    try:
        cursor = int(request.args.get("cursor", 0))
    except ValueError:
        cursor = 0
    with _JOB_LOCK:
        lines = _JOB["lines"]
        return jsonify({
            "stage": _JOB["stage"],
            "status": _JOB["status"],
            "returncode": _JOB["returncode"],
            "cursor": len(lines),
            "lines": lines[max(0, cursor):],
        })


@bp.route("/api/run/stop", methods=["POST"])
def api_run_stop():
    with _JOB_LOCK:
        proc = _JOB.get("proc")
        if proc and _JOB["status"] == "running":
            proc.terminate()
            _JOB["status"] = "stopped"
            _JOB["lines"].append("── stopped by user ──")
            return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "no running job"}), 409
