"""Test QuenLM OCR with webcam."""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ocr import QwenOCREngine


def capture_webcam_frame(cam_index: int = 0) -> np.ndarray:
    """
    Capture a single frame from the laptop webcam.

    Args:
    ----
    cam_index: Index of the webcam device to open.

    Returns:
    -------
    numpy.ndarray: RGB image array of shape HxWx3.

    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("ERROR: Could not open webcam.")

    print("Press SPACE to capture image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # show live preview
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)

        if key == 32:  # SPACE
            break
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")

    cap.release()
    cv2.destroyAllWindows()

    # convert BGR -> RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, prefix: str = "capture") -> Path:
    """Save the captured image for debugging."""
    out_path = Path(f"{prefix}_{datetime.now().strftime('%H-%M-%S')}.jpg")
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved captured image to: {out_path}")
    return out_path


def main():
    """Run an OCR test using the laptop webcam."""
    print("=== OCR Webcam Frame Test ===")

    rgb_img = capture_webcam_frame()
    save_image(rgb_img)

    print("Initializing OCR...")
    engine = QwenOCREngine(model_id="Qwen/Qwen3-VL-2B-Instruct")

    print("Running OCR...")
    result = engine.recognize(rgb_img)

    print("\n===== RAW MODEL OUTPUT =====")
    print(result.raw_output)

    print("\n===== CLEAN TEXT =====")
    print(result.text)

    print("\n===== LINES =====")
    for i, line in enumerate(result.lines):
        print(f"{i + 1}. {line}")


if __name__ == "__main__":
    main()
