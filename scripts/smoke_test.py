#!/usr/bin/env python3
"""GPU smoke test: ground a handful of instructions on the committed scene.

Useful for verifying that the model loads and that CUDA is reachable before
spending hours on the full sweep. The model path defaults to the HuggingFace
identifier; pass ``--model <local-or-hf-path>`` if you run fully offline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from PIL import Image

from instructscope.engine import GroundEngine
from instructscope.perturb import make_instruction

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scene", default="data/scenes/scene_seed0.png", type=Path)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no cuda available", file=sys.stderr)
        sys.exit(1)

    img = Image.open(args.scene).convert("RGB")
    eng = GroundEngine(args.model, max_pixels=448 * 448)

    cells = [
        ("canonical", "red", 0),
        ("ambiguous", "blue", 0),
        ("ambiguous", "green", 1),
        ("ambiguous", "yellow", 2),
        ("coref", "blue", 2),
    ]
    for family, color, variant in cells:
        inst = make_instruction(family, color, variant=variant, seed=0)
        resp = eng.ground(img, inst.text)
        pred = eng.parse_color(resp)
        print(f"{family}/{color}/v{variant}: '{inst.text}' -> pred={pred} ok={pred == color}", flush=True)

    print("SMOKE OK")


if __name__ == "__main__":
    main()
