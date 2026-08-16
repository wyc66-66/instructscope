"""InstructScope interactive dashboard.

Serves the sweep summary as a live UI: per-family reliability bars, the
saliency-bias fallback view, and the raw per-instruction table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from instructscope.analysis import load_sweep, summarize

ROOT = Path(__file__).resolve().parents[3]          # instructscope/
STATIC = ROOT / "ui" / "static"
assert STATIC.exists(), STATIC

app = FastAPI(title="InstructScope", description="Instruction perturbation reliability dashboard")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/summary")
def api_summary():
    sweep_path = ROOT / "data" / "sweep" / "sweep.json"
    if not sweep_path.exists():
        return JSONResponse({"error": "no sweep data found; run scripts/run_sweep.py first"}, status_code=404)
    data = load_sweep(sweep_path)
    return summarize(data)


@app.get("/sweep.json")
def sweep_json():
    sweep_path = ROOT / "data" / "sweep" / "sweep.json"
    if not sweep_path.exists():
        return JSONResponse({"error": "no sweep data found"}, status_code=404)
    return FileResponse(sweep_path)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
# The dashboard references its assets (vendor/chart.umd.min.js) relative to the
# page root — the same layout GitHub Pages serves. Mount the static directory at
# the root as a fallback so the local dashboard behaves identically to the
# deployed one.
app.mount("/", StaticFiles(directory=STATIC), name="static_root")
