"""Language->action grounding engine built on Qwen2.5-VL.

The policy is deliberately decomposed so the experiment isolates the
*instruction-to-object* mapping rather than conflating it with low-level
control:

- :meth:`ground` reads the agentview image + instruction and returns the
  predicted target colour. This is the semantic bridge we perturb.
- The downstream pick-and-place (locate -> move -> grasp -> lift) is a fixed
  scripted controller in :mod:`instructscope.rollout`, so a wrong target can
  only come from a failure of the semantic layer.

``ground`` asks the model to answer with a single colour name, which keeps the
interface constrained and makes the answer comparable to ground truth.
"""
from __future__ import annotations

import re

import torch
from PIL import Image

# The colours the environment can contain, in a fixed order for probing.
COLOR_NAMES = ("red", "blue", "green", "yellow", "purple", "orange")

_QUESTION = (
    "You are controlling a robot arm at a table. A user instruction tells you "
    "which object to pick up. Look at the image and decide which colour object "
    "the instruction refers to. Reply with ONLY the colour name (red, blue, "
    "green, yellow, purple or orange).\n\n"
    "Instruction: {instruction}"
)


class GroundEngine:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct", *, max_pixels: int = 448 * 448):
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.max_pixels = max_pixels

    @torch.inference_mode()
    def ground(self, image: Image.Image, instruction: str, *, max_new_tokens: int = 24) -> str:
        if image.width * image.height > self.max_pixels:
            scale = (self.max_pixels / (image.width * image.height)) ** 0.5
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        question = _QUESTION.format(instruction=instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        resp = self.processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        return resp

    @staticmethod
    def parse_color(response: str) -> str | None:
        """Extract a colour name from a model response, case-insensitively."""
        low = response.lower()
        for c in COLOR_NAMES:
            if c in low:
                return c
        return None
