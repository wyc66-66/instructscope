"""Instruction perturbation construction.

The core object is a *perturbation spectrum*: six families that take the same
underlying task (``pick up the red cube``) and stress a different part of the
language->action bridge:

- ``canonical``: the clean instruction, used as the reference success rate.
- ``paraphrase``: natural rewordings (synonyms, different syntax) that keep
  the exact same meaning.
- ``oov``: the colour name is replaced by an out-of-vocabulary but
  semantically-equivalent invented name (``vermilion`` for red, ``cobalt``
  for blue...). The words are unlikely to appear in the policy's training
  data, forcing composition from colour semantics.
- ``coref``: the target is referred to by pronoun / deixis (``it``,
  ``that one``, ``the one on the right``) which must be resolved from the
  scene.
- ``compound``: two sub-tasks joined by AND (``pick up the red cube then
  move it to the yellow region``), requiring instruction segmentation and
  sequencing.
- ``ambiguous``: the instruction underspecifies the target (``pick up a
  cube``), so the policy must pick a consistent default.

Every instruction is built from a fixed template bank so the sweep is fully
deterministic and the only moving part between families is the instruction
string itself.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# Colour vocabulary the policy must ground to the scene. ``oov`` variants are
# deliberately rare colour words that an instruction-tuned model may still
# understand but that would not appear in a vanilla tabletop corpus.
COLOR_OOV = {
    "red": "vermilion",
    "blue": "cobalt",
    "green": "viridian",
    "yellow": "amber",
    "purple": "violet",
    "orange": "tangerine",
}

FAMILIES = ("canonical", "paraphrase", "oov", "coref", "compound", "ambiguous")

_CANONICAL = {
    "red": "Pick up the red cube.",
    "blue": "Pick up the blue cube.",
    "green": "Pick up the green cube.",
    "yellow": "Pick up the yellow cube.",
}

_PARAPHRASE = {
    "red": [
        "Grab the red one.",
        "Please take the red block.",
        "Lift the red object off the table.",
        "Pick up the cube that is red.",
        "Could you get the red cube for me?",
    ],
    "blue": [
        "Grab the blue one.",
        "Please take the blue block.",
        "Lift the blue object off the table.",
        "Pick up the cube that is blue.",
        "Could you get the blue cube for me?",
    ],
    "green": [
        "Grab the green one.",
        "Please take the green block.",
        "Lift the green object off the table.",
        "Pick up the cube that is green.",
        "Could you get the green cube for me?",
    ],
    "yellow": [
        "Grab the yellow one.",
        "Please take the yellow block.",
        "Lift the yellow object off the table.",
        "Pick up the cube that is yellow.",
        "Could you get the yellow cube for me?",
    ],
}

_COREF = {
    "red": [
        "Pick up that cube.",
        "Grab it — the red one.",
        "Take the one that is leftmost.",
        "Pick up the first one you see that is red.",
    ],
    "blue": [
        "Pick up that cube.",
        "Grab it — the blue one.",
        "Take the one that is rightmost.",
        "Pick up the first one you see that is blue.",
    ],
    "green": [
        "Pick up that cube.",
        "Grab it — the green one.",
        "Take the one that is farthest from you.",
        "Pick up the first one you see that is green.",
    ],
    "yellow": [
        "Pick up that cube.",
        "Grab it — the yellow one.",
        "Take the one that is nearest to you.",
        "Pick up the first one you see that is yellow.",
    ],
}

_COMPOUND = {
    "red": [
        "Pick up the red cube and hold it steady.",
        "Lift the red cube, then do not put it down.",
        "Grasp the red cube and raise it above the table.",
    ],
    "blue": [
        "Pick up the blue cube and hold it steady.",
        "Lift the blue cube, then do not put it down.",
        "Grasp the blue cube and raise it above the table.",
    ],
    "green": [
        "Pick up the green cube and hold it steady.",
        "Lift the green cube, then do not put it down.",
        "Grasp the green cube and raise it above the table.",
    ],
    "yellow": [
        "Pick up the yellow cube and hold it steady.",
        "Lift the yellow cube, then do not put it down.",
        "Grasp the yellow cube and raise it above the table.",
    ],
}

_AMBIGUOUS = {
    "red": [
        "Pick up a cube.",
        "Pick up any cube you like.",
        "Choose a cube and pick it up.",
    ],
    "blue": [
        "Pick up a cube.",
        "Pick up any cube you like.",
        "Choose a cube and pick it up.",
    ],
    "green": [
        "Pick up a cube.",
        "Pick up any cube you like.",
        "Choose a cube and pick it up.",
    ],
    "yellow": [
        "Pick up a cube.",
        "Pick up any cube you like.",
        "Choose a cube and pick it up.",
    ],
}


@dataclass
class Instruction:
    family: str
    color: str
    text: str
    variant: int = 0
    meta: dict = field(default_factory=dict)


def make_instruction(family: str, color: str, variant: int = 0, seed: int = 0) -> Instruction:
    """Deterministically build one instruction for a family+color.

    ``variant`` selects among the template bank (per color where the family
    distinguishes colours). The ambiguous family's templates are colour-agnostic
    by design (the instruction must not name the target); with five sweep
    variants over a three-template bank, templates 0-1 are re-presented across
    all nominal targets, which measures the *determinism* of the fallback
    anchor rather than assuming it.
    """
    if family == "canonical":
        return Instruction(family, color, _CANONICAL[color], variant=0)
    if family == "paraphrase":
        bank = _PARAPHRASE[color]
        return Instruction(family, color, bank[variant % len(bank)], variant=variant)
    if family == "oov":
        oov = COLOR_OOV[color]
        text = f"Pick up the {oov} cube."
        return Instruction(family, color, text, variant=0, meta={"oov_word": oov})
    if family == "coref":
        bank = _COREF[color]
        return Instruction(family, color, bank[variant % len(bank)], variant=variant)
    if family == "compound":
        bank = _COMPOUND[color]
        return Instruction(family, color, bank[variant % len(bank)], variant=variant)
    if family == "ambiguous":
        bank = _AMBIGUOUS[color]
        idx = variant % len(bank)
        return Instruction(family, color, bank[idx], variant=idx)
    raise ValueError(f"unknown family {family!r}")
