# InstructScope

**Where instruction perturbations break open-vocabulary policy grounding.**

An empirical study of the *language grounding reliability boundary* of a
Qwen2.5-VL-grounded pick-and-place policy. The policy is probed across an
**instruction perturbation spectrum** — paraphrase, out-of-vocabulary words,
coreference, compound and ambiguous instructions — in a custom multi-object
robosuite environment.

The question is a direct companion to action-space reasoning in VLA policies.
ACoT-VLA (AgiBot, CVPR 2026) argues that the semantic-kinematic gap is best
bridged by deliberating in the *action* space rather than detouring through
language subtasks — but its evaluation, like every VLA today, measures
reliability over a fixed instruction set. That leaves the *language* side of the
bridge unmeasured: how far does open-vocabulary grounding actually reach, and
which instruction perturbations break it? This project measures exactly that —
the grounding reliability on which an action-space reasoner depends. If a policy
reasons in the action space but grounds the wrong object, the reasoning is
wasted.

## Headline finding

| Family | Example | Grounded success |
|---|---|---|
| canonical | `Pick up the red cube.` | 100% |
| paraphrase | `Please take the red block.` | 100% |
| out-of-vocab | `Pick up the vermilion cube.` | 100% |
| compound | `Grasp the red cube and raise it above the table.` | 100% |
| **coreference** | `Grab it — the red one.` | **55%** |
| **ambiguous** | `Pick up a cube.` | **25%** |

Lexical perturbations are fully absorbed; pragmatic perturbations are not.
Under underspecified instructions the fallback is systematically biased — in
the original layout it anchors to the two visually dominant cubes and the
exact anchor switches with the surface phrasing (χ² uniformity p = 7.9×10⁻⁵;
Fisher exact p = 7.9×10⁻⁶). The failures reveal a measurable **saliency
prior**, not noise.

**Scale:** 6 perturbation families × 4 colours × 5 variants × 4 layouts =
**480 deterministic cells** — the full 120-cell grid runs on the original
scene and on three colour-permuted re-arrangements. Every cell re-grounded
across the full perturbation bank with exact reproducibility (committed
scenes, one checkpoint).

**Cross-layout stability:** the 100%/55%/25% profile is layout-invariant
(Fisher exact p = 5.8×10⁻¹⁵ in each of the four layouts); the fallback anchor
moves with the layout; the original phrase-sensitivity does not transfer
(§3.4 of the report).

## Repository layout

```
instructscope/
├── src/instructscope/
│   ├── env/multilift.py     # MultiLift: robosuite env with N colour cubes
│   ├── perturb.py           # deterministic instruction perturbation bank
│   ├── engine.py            # Qwen2.5-VL grounding engine (image+text -> colour)
│   ├── rollout.py           # noise-free pick execution (semantic isolation)
│   ├── analysis.py          # Wilson intervals, saliency bias, Fisher exact
│   └── ui/app.py            # FastAPI dashboard
├── scripts/
│   ├── render_scenes.py     # render + cache the deterministic scene (CPU GL)
│   ├── run_sweep.py         # full perturbation sweep (checkpointed, resumable)
│   ├── render_figures.py    # paper figures
│   └── render_paper.py      # report.md -> report.html
├── ui/static/index.html     # interactive dashboard
├── docs/paper/report.md     # technical report
└── data/sweep/sweep.json    # raw per-cell records
```

## Reproducing the sweep

**Platform:** robosuite 1.4 supports Linux and macOS only (no native Windows).
On Windows use WSL2 with a working OpenGL setup (e.g. `apt install libegl1
mesa-utils`), or skip the sweep entirely — `data/sweep/` is committed and the
analysis, figures, and report reproduce from it on any OS.

Rendering uses CPU software GL (`LIBGL_ALWAYS_SOFTWARE=1`, llvmpipe) so the
MuJoCo renderer never touches WSL's unstable GPU virtualisation layer; the
VLM still runs on CUDA.

```bash
# 0. install (analysis needs numpy/scipy; the sweep needs robosuite + the VLM)
pip install -e ".[sim,gpu,paper,ui]"     # ~10 min; robosuite + torch
pip install pytest

# 0b. GPU smoke test (optional, ~1 min): verifies the engine loads and grounds
#     one instruction before you commit to the full sweep
python scripts/smoke_test.py

# 1. render the deterministic scene once (the canonical layout)
python scripts/render_scenes.py --out data/scenes

# 2. run the perturbation sweep on the canonical layout (120 cells; checkpointed
#    after every cell), then render and sweep the three colour-permuted scenes
TRANSFORMERS_OFFLINE=1 python scripts/run_sweep.py --out data/sweep
TRANSFORMERS_OFFLINE=1 python scripts/run_sweep.py --out data/sweep_v1
TRANSFORMERS_OFFLINE=1 python scripts/run_sweep.py --out data/sweep_v2
TRANSFORMERS_OFFLINE=1 python scripts/run_sweep.py --out data/sweep_v3
#    (480 cells total across the four layouts; see scripts/render_scenes.py --colors)

# 3. analyse + render figures (pass the variant sweeps for the cross-layout figure) + render the report
python scripts/cross_scene_facts.py --v0 data/sweep/sweep.json --v1 data/sweep_v1/sweep.json --v2 data/sweep_v2/sweep.json --v3 data/sweep_v3/sweep.json
python scripts/render_figures.py --sweep data/sweep/sweep.json --variants data/sweep_v1/sweep.json data/sweep_v2/sweep.json data/sweep_v3/sweep.json
python scripts/render_paper.py
```

> The `TRANSFORMERS_OFFLINE=1` / `LIBGL_ALWAYS_SOFTWARE=1` prefixes are bash
> syntax; on PowerShell (WSL) call the same commands inside `wsl bash -c '...'`.

## Tests

```bash
python -m pytest -q        # 10 tests: Wilson intervals, saliency analysis, coref split
```

CI (`.github/workflows/ci.yml`) runs the suite on every push to `main`.

## Dashboard

```bash
python -m uvicorn instructscope.ui.app:app --port 8733
# open http://localhost:8733
```

## Method

- **Model:** `Qwen/Qwen2.5-VL-3B-Instruct` (bf16, sdpa, offline)
- **Environment:** custom `MultiLift` built on robosuite 1.4.1 / MuJoCo 2.3.7
  — four colour-distinct cubes on a deterministic per-seed grid; agentview
  camera at 256×256
- **Isolation:** a grounded colour is executed noise-free (sim-state lift), so
  every failure is attributable to the semantic grounding layer, not motor
  control
- **Full grid:** 6 families × 4 colours × 5 variants = 120 deterministic cells
  per layout, re-run on four colour-permuted layouts (480 cells total; the
  pragmatic boundary is layout-invariant, §3.4 of the report)

---

## Live report

The technical report, figures and every number are served at **[https://wyc66-66.github.io/instructscope/](https://wyc66-66.github.io/instructscope/)**.
