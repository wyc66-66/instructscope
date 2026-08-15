"""instructscope: where instruction perturbations break open-vocabulary VLA policies.

We take a lightweight open VLA policy built on Qwen2.5-VL (visual perception +
instruction -> action plan, closed-loop in robosuite), and sweep an instruction
perturbation spectrum against task success. The spectrum covers paraphrase,
out-of-vocabulary terms, coreference, compound instructions and ambiguous
instructions. We locate the reliability boundary of the instruction space:
which perturbations are absorbed, and which quietly break the policy.
"""
