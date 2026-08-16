#!/usr/bin/env python3
"""Cross-scene stability analysis for the InstructScope perturbation spectrum.

The single-scene limitation of the main sweep is addressed by re-running the
identical 120-cell grid on three additional deterministic layouts (seeds 1-3).
This script re-derives the report's cross-scene claims from the four sweep
JSONs (data/sweep + data/sweep_s1..s3):

- per-family success across all scenes (is the lexical ceiling / pragmatic
  drop stable? is the *anchor* stable?)
- the ambiguous-family fallback distribution per scene (does the saliency
  prior survive a different layout?)
- the phrase->anchor switch per scene
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from instructscope.analysis import wilson_interval  # noqa: E402

SCENES = {
    "v0": "data/sweep/sweep.json",
    "v1": "data/sweep_v1/sweep.json",
    "v2": "data/sweep_v2/sweep.json",
    "v3": "data/sweep_v3/sweep.json",
}
# colour->position permutation per scene variant (the layout variable)
PERM = {
    "v0": "red→pos0 blue→pos1 green→pos2 yellow→pos3",
    "v1": "blue→pos0 green→pos1 yellow→pos2 red→pos3",
    "v2": "green→pos0 yellow→pos1 red→pos2 blue→pos3",
    "v3": "yellow→pos0 red→pos1 blue→pos2 green→pos3",
}
COLORS = ("red", "blue", "green", "yellow")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k, p in SCENES.items():
        ap.add_argument(f"--{k}", default=p, type=Path)
    args = ap.parse_args()

    sweeps = {k: json.loads(getattr(args, k).read_text(encoding="utf-8")) for k in SCENES}

    print("== per-family success per scene ==")
    fam_rows = {}
    for scene, data in sweeps.items():
        records = data["records"]
        by_fam = {}
        for r in records:
            by_fam.setdefault(r["family"], []).append(r)
        for fam, rs in by_fam.items():
            k = sum(r["success"] for r in rs)
            n = len(rs)
            by_fam[fam] = {"k": k, "n": n, "rate": k / n}
        fam_rows[scene] = by_fam
        print(f"  {scene}: " + "  ".join(
            f"{fam}={by_fam[fam]['rate']:.2f}" for fam in ("canonical", "paraphrase", "oov", "compound", "coref", "ambiguous")
        ))

    print("\n== lexical vs pragmatic boundary per scene ==")
    for scene, by_fam in fam_rows.items():
        lex = [by_fam[f]["k"] for f in ("canonical", "paraphrase", "oov", "compound")]
        lex_n = sum(by_fam[f]["n"] for f in ("canonical", "paraphrase", "oov", "compound"))
        prag = [by_fam[f]["k"] for f in ("coref", "ambiguous")]
        prag_n = sum(by_fam[f]["n"] for f in ("coref", "ambiguous"))
        # Fisher exact lexical vs pragmatic
        table = [[sum(lex), lex_n - sum(lex)], [sum(prag), prag_n - sum(prag)]]
        _, p = stats.fisher_exact(table)
        print(f"  {scene}: lexical {sum(lex)}/{lex_n} ({sum(lex)/lex_n:.1%}) vs "
              f"pragmatic {sum(prag)}/{prag_n} ({sum(prag)/prag_n:.1%})  Fisher p={p:.2e}")

    print("\n== ambiguous fallback distribution per scene (saliency prior) ==")
    for scene, data in sweeps.items():
        amb = [r for r in data["records"] if r["family"] == "ambiguous"]
        counts = {c: 0 for c in COLORS}
        for r in amb:
            if r["predicted"]:
                counts[r["predicted"]] += 1
        n = len(amb)
        # chi-square vs uniform
        chi2 = sum((counts[c] - n / len(COLORS)) ** 2 / (n / len(COLORS)) for c in COLORS)
        up = float(stats.chi2.sf(chi2, df=len(COLORS) - 1))
        # per-phrase anchor
        by_phrase = {}
        for r in amb:
            by_phrase.setdefault(r["instruction"], []).append(r["predicted"] or "")
        anchors = {}
        for phrase, preds in by_phrase.items():
            anchor = max(set(preds), key=preds.count)
            anchors[phrase] = (anchor, preds.count(anchor))
        print(f"  {scene}: counts={counts}  uniform_p={up:.2e}")
        for phrase, (a, c) in anchors.items():
            print(f"      '{phrase}' -> {a} ({c})")

    print("\n== coref mechanisms per scene ==")
    for scene, data in sweeps.items():
        coref = [r for r in data["records"] if r["family"] == "coref"]
        vac = [r for r in coref if "that cube" in r["instruction"].lower()]
        if vac:
            k = sum(r["success"] for r in vac)
            anchors = sorted({r["predicted"] for r in vac})
            print(f"  {scene}: vacuous-deixis {k}/{len(vac)} ({k/len(vac):.1%}) anchors={anchors}")
        else:
            print(f"  {scene}: no vacuous-deixis records")


if __name__ == "__main__":
    main()
