"""Use QuenLM to read writing on board."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


@dataclass
class OCRResult:
    """Container for OCR inference results."""

    text: str
    """Full transcribed text."""
    lines: list[str]
    """Text split into lines."""
    raw_output: str
    """Raw model output (for debugging)."""


# ------------ Begin_Citation [2] -------------
class QwenOCREngine:
    """
    Qwen3-VL-based OCR engine.

    - Input: single image (whiteboard crop / rectified board).
    - Output: plain text transcription.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the Qwen OCR engine.

        Args:
        ----
        model_id:
            HuggingFace model identifier for the Qwen3-VL variant to load.
        device:
            Device to run inference on ("cuda" or "cpu"). If None,
            the method automatically selects GPU when available.

        """
        if device is None:
            if torch.cuda.is_available():
                print("Using GPU for OCR")
                self._device = "cuda"
            else:
                print("Using CPU for OCR")
                self._device = "cpu"

        self._device = device
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto" if device == "cuda" else None,
        )
        self._processor = AutoProcessor.from_pretrained(model_id)

    @staticmethod
    def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
        """
        Convert a numpy RGB array into a PIL image if necessary.

        Args:
        ----
        image:
            Either a PIL.Image or an HxWx3 uint8 numpy array
            representing an RGB image.

        Returns:
        -------
            A valid PIL.Image object ready for processing.

        """
        if isinstance(image, Image.Image):
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return Image.fromarray(image.astype("uint8"))
        raise ValueError("Expected HxWx3 RGB numpy array or PIL.Image")

    def recognize(self, image: np.ndarray | Image.Image) -> OCRResult:
        """
        Run OCR on a single image and return a text transcription.

        Args:
        ----
        image:
            Rectified RGB image of the writing surface (HxWx3, uint8 or PIL).

        Returns:
        -------
        OCRResult:
            Structured output containing the cleaned text,
            line-separated text, and the raw model output.

        """
        pil_img = self._to_pil(image)

        # prompt it to behave like strict OCR
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {
                        "type": "text",
                        "text": (
                            "You are an OCR engine. "
                            "Read the handwritten text on this whiteboard and "
                            "transcribe it exactly as plain text, line by line. "
                            "Do not add explanations or comments."
                        ),
                    },
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
        # strip the prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_texts = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
# -------------- End_Citation [2] --------------

        raw = output_texts[0]

        # normalize newlines a bit
        text = raw.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        return OCRResult(text=text, lines=lines, raw_output=raw)
