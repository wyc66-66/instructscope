"""Statistical analysis of the instruction perturbation sweep.

Turns the raw per-cell records into per-family success estimates with Wilson
95% intervals, plus a saliency-bias table for the ambiguous family (which
object does the model fall back to when the instruction does not specify one).
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

from scipy import stats


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_sweep(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def summarize(data: dict) -> dict:
    records = data["records"]
    families = data["meta"]["families"]

    per_family = []
    for fam in families:
        rs = [r for r in records if r["family"] == fam]
        k = sum(r["success"] for r in rs)
        n = len(rs)
        lo, hi = wilson_interval(k, n)
        per_family.append(
            {
                "family": fam,
                "n": n,
                "success": k,
                "rate": k / n if n else 0.0,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )

    # Saliency bias: for ambiguous instructions, which colour does the model
    # fall back to when nothing in the instruction constrains the target?
    saliency = {}
    for r in records:
        if r["family"] == "ambiguous" and r["predicted"]:
            saliency[r["predicted"]] = saliency.get(r["predicted"], 0) + 1

    # Fisher exact: is the ambiguous fallback distribution different from
    # uniform across colours? (2x2: most-chosen colour vs the rest)
    most_colour, most_count = max(saliency.items(), key=lambda kv: kv[1])
    n_amb = sum(saliency.values())
    other = n_amb - most_count
    n_colours = len(data["meta"]["colors"])
    # expected under uniformity: n_amb / n_colours per colour
    table = [[most_count, n_amb - most_count], [1, n_colours - 1]]
    _, fisher_p = stats.fisher_exact(table)

    # Confusion: for each target colour, what did the model predict (all fams)
    confusion = {}
    for c in data["meta"]["colors"]:
        counts = {}
        for r in records:
            if r["color"] == c and r["predicted"]:
                counts[r["predicted"]] = counts.get(r["predicted"], 0) + 1
        confusion[c] = counts

    return {
        "meta": data["meta"],
        "per_family": per_family,
        "saliency": saliency,
        "saliency_most": {"colour": most_colour, "count": most_count, "total": n_amb},
        "fisher_p": float(fisher_p),
        "confusion": confusion,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="data/sweep/sweep.json", type=Path)
    ap.add_argument("--out", default="data/sweep/summary.json", type=Path)
    args = ap.parse_args()
    data = load_sweep(args.sweep)
    summary = summarize(data)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for f in summary["per_family"]:
        print(f"{f['family']:<12} {f['rate']:>7.1%}  [{f['ci_lo']:.1%}, {f['ci_hi']:.1%}]")
    print(f"saliency most: {summary['saliency_most']}  fisher_p={summary['fisher_p']:.4f}")
