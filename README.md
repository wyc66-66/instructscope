# InstructScope

**Where instruction perturbations break open-vocabulary VLA policies.**

An empirical study of the *language grounding reliability boundary* of an
open-vocabulary vision-language-action policy. A Qwen2.5-VL-grounded
pick-and-place policy is probed across an **instruction perturbation
spectrum** — paraphrase, out-of-vocabulary words, coreference, compound and
ambiguous instructions — in a custom multi-object robosuite environment.

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
Under underspecified instructions the policy falls back to a single visually
salient colour with perfect consistency — a measurable **saliency bias**
inherited from its training distribution (Fisher exact p < 0.01).

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

Rendering uses CPU software GL (`LIBGL_ALWAYS_SOFTWARE=1`, llvmpipe) so the
MuJoCo renderer never touches WSL's unstable GPU virtualisation layer; the
VLM still runs on CUDA.

```bash
# 1. render the deterministic scene once
python scripts/render_scenes.py --out data/scenes

# 2. run the perturbation sweep (120 cells; checkpointed after every cell)
TRANSFORMERS_OFFLINE=1 python scripts/run_sweep.py --out data/sweep

# 3. analyse + render figures + render the report
python scripts/render_figures.py --sweep data/sweep/sweep.json
python scripts/render_paper.py
```

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

## Citation

If you use this in your work, cite it as a research artifact:

```
Dong Hao (2026). InstructScope: Where instruction perturbations break
open-vocabulary VLA policies. Technical report.
```
---

## Live report

The technical report, figures and every number are served at **[https://wyc66-66.github.io/instructscope/](https://wyc66-66.github.io/instructscope/)**.
