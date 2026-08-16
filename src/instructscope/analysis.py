"""Statistical analysis of the instruction perturbation sweep.

Turns the raw per-cell records into per-family success estimates with Wilson
95% intervals, plus two behavioural analyses of the failures:

- ``default_analysis.ambiguous`` — the fallback distribution under fully
  unconstrained instructions, with a goodness-of-fit test against uniform,
  a test that all fallbacks land in the same two-colour set, and a per-phrase
  anchor breakdown (the anchor is deterministic *within* a phrase but switches
  across phrases).
- ``default_analysis.coref`` — the two failure mechanisms behind the 55%
  coreference rate (vacuous deixis vs spatial-reference error).

The schema is consumed by the interactive dashboard and the figure scripts.
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


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Two-proportion z-test between independent binomial groups.

    Used to ask whether the *observed levels* of two families (e.g.
    coreference 55% vs ambiguous 25%) differ, beyond their overlapping Wilson
    intervals. The two-sided p-value is from the normal approximation.
    """
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    p_pool = (k1 + k2) / (n1 + n2) if (n1 + n2) else 0.0
    se = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"z": float(z), "p": float(p), "p1": float(p1), "p2": float(p2),
            "n1": n1, "n2": n2}


def load_sweep(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ambiguous_analysis(records: list[dict], colours: list[str]) -> dict:
    """Fallback behaviour under fully unconstrained instructions."""
    amb = [r for r in records if r["family"] == "ambiguous"]
    counts = {c: 0 for c in colours}
    for r in amb:
        if r["predicted"]:
            counts[r["predicted"]] += 1
    n = len(amb)
    n_grounded = sum(counts.values())

    most_colour, most_count = max(counts.items(), key=lambda kv: kv[1])

    # 1) Is the fallback distribution uniform across colours?
    #    chi-square goodness-of-fit vs 20/4 = 5 expected per colour.
    chi2 = sum((counts[c] - n / len(colours)) ** 2 / (n / len(colours)) for c in colours)
    uniform_p = float(stats.chi2.sf(chi2, df=len(colours) - 1))

    # 2) All fallbacks land in a two-colour set (the two back-row cubes in
    #    this scene). Under a uniform prior the probability is 0.5**n.
    top2 = [c for c, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:2]
    top2_count = sum(counts[c] for c in top2)
    two_colour_p = float(0.5 ** n)

    # 3) Phrase -> anchor: the same under-specified intent, rephrased, anchors
    #    to a *different* colour deterministically. Group cells by instruction
    #    string and read off the per-phrase anchor.
    by_phrase: dict[str, list[str]] = {}
    for r in amb:
        by_phrase.setdefault(r["instruction"], []).append(r["predicted"] or "")
    phrase_anchor = []
    for phrase in sorted(by_phrase, key=lambda p: -len(by_phrase[p])):
        preds = by_phrase[phrase]
        anchor_counts = {c: preds.count(c) for c in colours}
        anchor = max(anchor_counts, key=lambda c: anchor_counts[c])
        phrase_anchor.append(
            {
                "phrase": phrase,
                "n": len(preds),
                "anchor": anchor,
                "count": anchor_counts[anchor],
            }
        )

    # Fisher exact: is the anchor colour independent of the surface form?
    # Collapse to the two observed anchor colours {a, b}; table rows = phrases
    # whose anchor is a vs b, columns = whether each cell fell on colour a or b.
    a, b = phrase_anchor[0]["anchor"], phrase_anchor[1]["anchor"]
    ga, gb = [p for p in phrase_anchor if p["anchor"] == a], [p for p in phrase_anchor if p["anchor"] == b]
    table = [
        [sum(p["count"] for p in ga), sum(p["n"] - p["count"] for p in ga)],
        [sum(p["n"] - p["count"] for p in gb), sum(p["count"] for p in gb)],
    ]
    _, phrase_p = stats.fisher_exact(table)

    return {
        "n": n,
        "n_grounded": n_grounded,
        "counts": counts,
        "most": {"colour": most_colour, "count": most_count},
        "uniform_p": uniform_p,
        "two_colour": {"colours": top2, "count": top2_count, "p": two_colour_p},
        "phrase_anchor": phrase_anchor,
        "phrase_anchor_p": float(phrase_p),
    }


def _coref_analysis(records: list[dict]) -> dict:
    """Break the coreference family down into its two failure mechanisms.

    The template bank mixes *vacuous deixis* ("pick up that cube", no
    antecedent) with *spatial reference* ("the one that is rightmost"), so
    lumping them hides two different failure modes. We split on the surface
    form to keep the report honest about what actually fails.
    """
    coref = [r for r in records if r["family"] == "coref"]

    def split(r):
        t = r["instruction"].lower()
        if "that cube" in t:
            return "vacuous"
        if any(w in t for w in ("leftmost", "rightmost", "nearest", "farthest")):
            return "spatial"
        if "first one you see" in t:
            return "attribute search"
        return "appositive" if "— the" in r["instruction"] else "other"

    groups: dict[str, list] = {}
    for r in coref:
        groups.setdefault(split(r), []).append(r)

    out = {"n": len(coref), "mechanisms": {}}
    for name, rs in groups.items():
        k = sum(r["success"] for r in rs)
        lo, hi = wilson_interval(k, len(rs))
        out["mechanisms"][name] = {
            "n": len(rs),
            "success": k,
            "rate": k / len(rs),
            "ci_lo": lo,
            "ci_hi": hi,
            "anchors": sorted({r["predicted"] for r in rs}),
        }
        if name == "vacuous":
            vac = [r["predicted"] for r in rs if r["predicted"]]
            if vac:
                most = max(set(vac), key=vac.count)
                out["vacuous_anchor"] = {"colour": most, "count": vac.count(most)}
    return out


def summarize(data: dict) -> dict:
    records = data["records"]
    colours = data["meta"]["colors"]
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

    confusion = {}
    for c in colours:
        counts = {}
        for r in records:
            if r["color"] == c and r["predicted"]:
                counts[r["predicted"]] = counts.get(r["predicted"], 0) + 1
        confusion[c] = counts

    # Legacy flat keys, kept for scripts that consumed the first schema.
    saliency = {c: 0 for c in colours}
    for r in records:
        if r["family"] == "ambiguous" and r["predicted"]:
            saliency[r["predicted"]] += 1
    most_colour, most_count = max(saliency.items(), key=lambda kv: kv[1])

    amb = _ambiguous_analysis(records, colours)
    coref = _coref_analysis(records)

    # Two-proportion test: are the observed levels of coreference vs ambiguous
    # grounding distinguishable, or just sampling noise around overlapping CIs?
    by_fam = {f["family"]: f for f in per_family}
    pair = None
    if "coref" in by_fam and "ambiguous" in by_fam:
        pair = two_proportion_z(
            by_fam["coref"]["success"], by_fam["coref"]["n"],
            by_fam["ambiguous"]["success"], by_fam["ambiguous"]["n"],
        )

    return {
        "meta": data["meta"],
        "per_family": per_family,
        "default_analysis": {"ambiguous": amb, "coref": coref},
        "confusion": confusion,
        "saliency": saliency,
        "saliency_most": {
            "colour": most_colour,
            "count": most_count,
            "total": sum(saliency.values()),
        },
        "fisher_p": amb["phrase_anchor_p"],
        "coref_vs_ambiguous": pair,
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
    amb = summary["default_analysis"]["ambiguous"]
    print(f"\nambiguous: n={amb['n']}, fallbacks={amb['counts']}, uniform_p={amb['uniform_p']:.2e}")
    print(f"  two-colour set {amb['two_colour']['colours']}: {amb['two_colour']['count']}/{amb['n']}, p={amb['two_colour']['p']:.2e}")
    print(f"  phrase_anchor_p={amb['phrase_anchor_p']:.2e}")
    for pp in amb["phrase_anchor"]:
        print(f"  '{pp['phrase']}' -> {pp['anchor']} ({pp['count']}/{pp['n']})")
    coref = summary["default_analysis"]["coref"]
    print("\ncoref mechanisms:")
    for name, m in coref["mechanisms"].items():
        print(f"  {name:<11} {m['rate']:>7.1%} ({m['success']}/{m['n']}) anchors={m['anchors']}")
