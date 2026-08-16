#!/usr/bin/env python3
"""Sweep the instruction perturbation spectrum against grounding success.

For every (family, colour, variant) cell the script:
  1. loads the cached agentview image of the deterministic MultiLift scene
  2. asks the VLM to ground the instruction to a colour
  3. records whether the grounded colour matches the intended target

Execution is deliberately noise-free (a grounded colour == a lifted cube), so
the metric isolates the instruction-to-object semantic bridge. The renderer
and the VLM run in separate processes: this script never imports mujoco, and
rendering happens once up-front in :mod:`render_scenes`.

Results are written as ``sweep.json`` with a ``meta`` block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from instructscope.engine import GroundEngine
from instructscope.perturb import FAMILIES, make_instruction

COLORS = ("red", "blue", "green", "yellow")
VARIANT_COUNT = 5


def run_cell(engine, image, color, family, variant, seed):
    inst = make_instruction(family, color, variant=variant, seed=seed)
    t0 = time.time()
    response = engine.ground(image, inst.text)
    predicted = engine.parse_color(response)
    grounding_time = time.time() - t0

    return {
        "family": family,
        "color": color,
        "variant": variant,
        "seed": seed,
        "instruction": inst.text,
        "response": response,
        "predicted": predicted,
        "target": color,
        "success": bool(predicted is not None and predicted == color),
        "grounding_time": round(grounding_time, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--colors", nargs="*", default=list(COLORS))
    ap.add_argument("--variants", type=int, default=VARIANT_COUNT)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default="data/sweep", type=Path)
    ap.add_argument("--scene", default="data/scenes/scene_seed0.png", type=Path)
    ap.add_argument("--resume", action="store_true",
                    help="skip records already present in the output file")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "sweep.json"

    records: list[dict] = []
    if args.resume and out_file.exists():
        records = json.loads(out_file.read_text(encoding="utf-8"))["records"]
        print(f"[sweep] resumed with {len(records)} existing records", flush=True)

    image = Image.open(args.scene).convert("RGB")
    scene_sha = hashlib.sha256(Path(args.scene).read_bytes()).hexdigest()
    engine = GroundEngine(args.model)

    done = len(records)
    seen = {(r["family"], r["color"], r["variant"], r["seed"]) for r in records}
    total = len(args.families) * len(args.colors) * args.variants * len(args.seeds)
    for family in args.families:
        for color in args.colors:
            for variant in range(args.variants):
                for seed in args.seeds:
                    key = (family, color, variant, seed)
                    if key in seen:
                        continue
                    rec = run_cell(engine, image, color, family, variant, seed)
                    records.append(rec)
                    done += 1
                    print(f"[{done}/{total}] {family}/{color}/v{variant}/s{seed}: "
                          f"pred={rec['predicted']} ok={rec['success']}", flush=True)

                    # checkpoint after every cell so a crash costs nothing
                    result = {
                        "meta": {
                            "model": args.model,
                            "colors": list(args.colors),
                            "families": list(args.families),
                            "variants": args.variants,
                            "seeds": list(args.seeds),
                            "image_size": image.size,
                            "scene": str(args.scene),
                            "scene_sha256": scene_sha,
                        },
                        "records": records,
                    }
                    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[sweep] wrote {len(records)} records -> {out_file}")


if __name__ == "__main__":
    main()
