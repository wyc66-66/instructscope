"""Tests for InstructScope perturbation analysis."""
from __future__ import annotations

import json

import pytest

from instructscope.analysis import (
    _ambiguous_analysis,
    _coref_analysis,
    summarize,
    two_proportion_z,
    wilson_interval,
)


class TestWilsonInterval:
    def test_zero_n(self):
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_contains_p(self):
        lo, hi = wilson_interval(11, 20)
        assert lo <= 0.55 <= hi

    def test_wide_at_small_n(self):
        # report quotes coref 55% = [34.2%, 74.2%]
        lo, hi = wilson_interval(11, 20)
        assert abs(lo - 0.342) < 0.01
        assert abs(hi - 0.742) < 0.01

    def test_p1_narrow(self):
        lo, hi = wilson_interval(200, 200)
        assert lo > 0.97


def _record(family, color, success, predicted, instruction=""):
    return {
        "family": family,
        "color": color,
        "success": success,
        "predicted": predicted,
        "instruction": instruction,
    }


class TestAmbiguousAnalysis:
    def _records(self):
        # blue-anchored phrase, yellow-anchored phrase, uniform-ish mix
        recs = []
        for _ in range(12):
            recs.append(_record("ambiguous", "red", 0, "blue", "pick up a cube"))
        for _ in range(8):
            recs.append(_record("ambiguous", "red", 0, "yellow", "pick up any cube"))
        return recs

    def test_two_colour_claim(self):
        amb = _ambiguous_analysis(self._records(), ["blue", "yellow", "red", "green"])
        assert amb["n"] == 20
        assert amb["two_colour"]["count"] == 20  # all fallbacks in top-2 set
        assert amb["two_colour"]["p"] == 0.5 ** 20

    def test_phrase_anchor_switch(self):
        amb = _ambiguous_analysis(self._records(), ["blue", "yellow", "red", "green"])
        anchors = {p["phrase"]: p["anchor"] for p in amb["phrase_anchor"]}
        assert anchors["pick up a cube"] == "blue"
        assert anchors["pick up any cube"] == "yellow"
        assert amb["phrase_anchor_p"] < 1e-3

    def test_uniform_null(self):
        # each colour grounded by its own phrase -> near-uniform distribution
        phrases = ["pick up the blue one", "pick up the yellow one",
                   "pick up the red one", "pick up the green one"]
        recs = [_record("ambiguous", c, 0, c, phrases[i])
                for i, c in enumerate(["blue", "yellow", "red", "green"]) for _ in range(5)]
        amb = _ambiguous_analysis(recs, ["blue", "yellow", "red", "green"])
        assert amb["uniform_p"] > 0.1


class TestCorefAnalysis:
    def test_mechanism_split(self):
        recs = [
            _record("coref", "blue", 1, "blue", "Grab it — the blue one."),
            _record("coref", "red", 0, "blue", "pick up that cube"),
            _record("coref", "red", 0, "blue", "pick up that cube"),
            _record("coref", "green", 1, "green", "Take the one that is rightmost."),
        ]
        c = _coref_analysis(recs)
        assert c["n"] == 4
        mech = c["mechanisms"]
        assert mech["appositive"]["rate"] == 1.0
        assert mech["vacuous"]["rate"] == 0.0
        assert mech["spatial"]["rate"] == 1.0
        assert c["vacuous_anchor"]["colour"] == "blue"

    def test_vacuous_anchor_consistent(self):
        recs = [_record("coref", "red", 0, "yellow", "pick up that cube") for _ in range(6)]
        c = _coref_analysis(recs)
        assert c["vacuous_anchor"] == {"colour": "yellow", "count": 6}


class TestSummarize:
    def test_per_family_rates(self):
        data = {
            "meta": {"colors": ["blue", "yellow", "red", "green"], "families": ["canonical", "ambiguous"]},
            "records": [
                _record("canonical", "blue", 1, "blue") for _ in range(20)
            ]
            + [_record("ambiguous", "blue", 0, "blue", "pick up a cube") for _ in range(10)]
            + [_record("ambiguous", "yellow", 0, "yellow", "pick up any cube") for _ in range(10)],
        }
        s = summarize(data)
        pf = {f["family"]: f for f in s["per_family"]}
        assert pf["canonical"]["rate"] == 1.0
        assert pf["ambiguous"]["rate"] == 0.0
        assert s["fisher_p"] < 1.0
        assert "default_analysis" in s
        # no coref family in this fixture -> pairwise test absent
        assert s["coref_vs_ambiguous"] is None


class TestTwoProportion:
    def test_significant_difference(self):
        # 55% (11/20) vs 25% (5/20): not significant at alpha=0.05, near threshold
        r = two_proportion_z(11, 20, 5, 20)
        assert r["p1"] == 0.55
        assert r["p2"] == 0.25
        assert r["p"] > 0.05

    def test_equal_proportions(self):
        r = two_proportion_z(10, 20, 10, 20)
        assert r["z"] == 0.0
        assert r["p"] == 1.0

    def test_extreme_split(self):
        # 16/40 vs 80/80: the report's pooled boundary, Fisher-verified ~6e-15
        r = two_proportion_z(16, 40, 80, 80)
        assert r["p"] < 1e-12
