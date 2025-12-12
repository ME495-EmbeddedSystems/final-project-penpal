"""Gemini OCR + QA Engine."""
from dataclasses import dataclass
import io
import json
import os
from typing import Optional

from google import genai
from google.genai import types
import numpy as np
from PIL import Image


@dataclass
class OCRResult:
    """Container for OCR inference results."""

    text: str
    """Full transcribed text."""
    lines: list[str]
    """Text split into lines."""
    raw_output: str
    """Raw model output (for debugging)."""


@dataclass
class BoardQAResult:
    """Container for full board pipeline (OCR + answer)."""

    question: str
    """Question text read from the board (currently the full OCR text)."""
    answer: str
    """Model-generated answer text."""
    ocr: OCRResult
    """Full OCR result (text, lines, raw output)."""
    raw_answer_output: str
    """Raw model output for the answer call (for debugging)."""


class GeminiOCREngine:
    """
    Gemini-based OCR engine using the 'google-genai' SDK.

    - Input: single image (whiteboard crop / rectified board) and/or text.
    - Output:
        * OCR: text transcription of board content.
        * QA: text answer using the same vision-language model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = 'gemini-2.0-flash',
    ) -> None:
        """
        Initialize the Gemini OCR engine.

        Args:
        ----
        api_key:
            API key for Google Gemini access. If None, reads from
            'GOOGLE_API_KEY' environment variable.
        model_name:
            Gemini model name to use for OCR and QA.

        """
        self.api_key = (
            api_key
            or os.getenv('GOOGLE_API_KEY')
        )

        if not self.api_key:
            raise ValueError(
                'No API key provided. Set GOOGLE_API_KEY env var.'
            )

        self._client = genai.Client(api_key=self.api_key)
        self._model_name = model_name

    @staticmethod
    def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                return Image.fromarray(image.astype('uint8'))
        raise ValueError('Expected HxWx3 RGB numpy array or PIL.Image')

    def _image_to_part(self, image: Image.Image) -> types.Part:
        """Convert PIL image to Gemini Part (Byte handling)."""
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return types.Part.from_bytes(
            data=img_bytes.read(), mime_type='image/png'
        )

    def read_and_answer_board(
        self,
        image: np.ndarray | Image.Image,
        context: Optional[str] = None,
    ) -> BoardQAResult:
        """OCR the board and generate an answer."""
        pil_img = self._to_pil(image)
        image_part = self._image_to_part(pil_img)

        prompt_text = (
            'Analyze this whiteboard image.\n'
            '1. Transcribe exactly what is written on the board.\n'
            '2. Treat the transcription as a question and'
            '   provide a clear, concise answer.\n'
            'Return the result as a valid JSON object with this schema:\n'
            '{\n'
            '  "transcription": "string (exact text on board)",\n'
            '  "answer": "string (your response to the text)"\n'
            '}'
        )

        text_part = types.Part.from_text(text=prompt_text)

        # JSON response type
        config = types.GenerateContentConfig(
            response_mime_type='application/json', temperature=0.0
        )

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[types.Content(parts=[image_part, text_part])],
                config=config,
            )

            raw_text = response.text
            data = json.loads(raw_text)

            question_text = data.get('transcription', '')
            answer_text = data.get('answer', '')

            lines = [
                ln.strip() for ln in question_text.splitlines() if ln.strip()
            ]

            ocr_res = OCRResult(
                text=question_text, lines=lines, raw_output=raw_text
            )

            return BoardQAResult(
                question=question_text,
                answer=answer_text,
                ocr=ocr_res,
                raw_answer_output=raw_text,
            )

        except Exception as e:
            print(f'Gemini QA Error: {e}')
            empty_ocr = OCRResult(text='', lines=[], raw_output=str(e))
            return BoardQAResult(
                question='',
                answer=f'Error processing board: {e}',
                ocr=empty_ocr,
                raw_answer_output=str(e),
            )
