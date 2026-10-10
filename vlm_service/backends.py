"""Vision-language model backends for the VLM service.

A backend answers a question about a short sequence of camera frames: the last image is the current
frame, the ones before it are earlier frames, oldest first.
"""
import warnings
from abc import ABC, abstractmethod

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SYSTEM_PROMPT = ("You are a camera assistant. You see frames from a camera and answer questions about what "
                 "the camera sees. Earlier frames show what happened before the current frame. "
                 "Answer briefly, in 1-2 sentences.")


class VLMBackend(ABC):
    """Interface of a vision-language model backend."""

    name: str  # model id, stored with every answer

    @abstractmethod
    def answer(self, images: list[Image.Image], question: str) -> str:
        """Answers `question` about `images`; the last image is the current frame."""


class QwenVLBackend(VLMBackend):
    """Qwen2.5-VL on a CUDA GPU in bfloat16, following the model card (Qwen/Qwen2.5-VL-3B-Instruct)."""

    def __init__(self, model_id: str, max_pixels: int, max_new_tokens: int):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available, but the VLM service needs an NVIDIA GPU. Check that nvidia-smi "
                             "works and that the CUDA build of PyTorch is installed (vlm_service/requirements.txt).")
        self.name = model_id
        self.max_new_tokens = max_new_tokens
        # max_pixels caps the size of every image: larger images are scaled down to fit (in steps of 28 pixels).
        # Fewer pixels mean fewer visual tokens, so less memory and faster answers. The model's own processor
        # config also sets max_pixels, which transformers 5.19 reports as deprecated on every call; the cap
        # still applies, so that warning is hidden.
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        warnings.filterwarnings("ignore", message="Passing `min_pixels` and `max_pixels`", category=FutureWarning)
        # bfloat16, never float32: the 3B model takes about 7 GB in bfloat16 but twice that in float32,
        # which would not fit the 12 GB GPU.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda").eval()

    @staticmethod
    def build_messages(image_count: int, question: str) -> list:
        """Chat messages: the system prompt, then the earlier frames (oldest to newest), the current
        frame and the question. {"type": "image"} marks where each image goes."""
        content = []
        if image_count > 1:
            content.append({"type": "text", "text": "Earlier frames, from oldest to newest:\n"})
            for number in range(1, image_count):
                content += [{"type": "text", "text": f"Earlier frame {number}:"}, {"type": "image"},
                            {"type": "text", "text": "\n"}]
        content += [{"type": "text", "text": "Current frame:"}, {"type": "image"},
                    {"type": "text", "text": f"\n{question}"}]
        return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content}]

    def answer(self, images: list[Image.Image], question: str) -> str:
        prompt = self.processor.apply_chat_template(self.build_messages(len(images), question),
                                                    tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)  # greedy
        new_tokens = output[:, inputs["input_ids"].shape[1]:]  # drop the prompt
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
