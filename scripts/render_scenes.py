#!/usr/bin/env python3
"""Render the deterministic MultiLift scene once and cache it as a PNG.

The scene layout is a fixed function of (colors, seed); the sweep can then run
entirely from the cached image, decoupling the renderer from the VLM process.
Rendering runs on the CPU (llvmpipe) to avoid WSL's unstable GPU path.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from instructscope.env import build_env

COLORS = ("red", "blue", "green", "yellow")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colors", nargs="*", default=list(COLORS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="data/scenes", type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    env = build_env(args.colors, target=args.colors[0], seed=args.seed)
    img = env.sim.render(args.size, args.size, camera_name="agentview")
    out_file = args.out / f"scene_seed{args.seed}.png"
    Image.fromarray(img).save(out_file)
    print(f"[render] wrote {out_file} mean={img.mean():.1f}")


if __name__ == "__main__":
    main()
