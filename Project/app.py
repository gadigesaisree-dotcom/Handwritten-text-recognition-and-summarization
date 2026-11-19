import os
import sys
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr
import cv2
import torch
import numpy as np
from PIL import Image

# Import custom modules
from src.preprocessing import preprocess_image, segment_lines, get_word_bounding_boxes
from src.htr_models import build_trocr
from src.postprocessing import TextPostProcessor
from src.summarization import TextSummarizer

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

print("Initializing models. This may take a minute...")
processor, trocr_model = build_trocr()
device = "cuda" if torch.cuda.is_available() else "cpu"
trocr_model.to(device)

post_processor = TextPostProcessor()
summarizer = TextSummarizer(config_path=_CONFIG_PATH)



_MIN_INK_RATIO = 0.005  
_MIN_LINE_HEIGHT = 10

# Minimum number of real words in a TrOCR output for the line to be kept.
# Single-character or all-symbol outputs are discarded.
_MIN_WORD_LEN = 2


def _is_real_text(ocr_output: str) -> bool:
    text = ocr_output.strip()
    if not text:
        return False

    words = re.findall(r"[A-Za-z]{%d,}" % _MIN_WORD_LEN, text)
    if not words:
        return False

    alpha_count = sum(c.isalpha() for c in text)
    total_printable = sum(c.isprintable() and not c.isspace() for c in text)
    if total_printable == 0 or (alpha_count / total_printable) < 0.30:
        return False

    return True


def _line_has_ink(binary_line: np.ndarray) -> bool:
    if binary_line is None or binary_line.size == 0:
        return False
    ink_ratio = np.count_nonzero(binary_line) / binary_line.size
    return ink_ratio >= _MIN_INK_RATIO


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_pipeline(image_path):
    import traceback

    if isinstance(image_path, dict):
        image_path = image_path.get("path") or image_path.get("name") or image_path.get("url")

    if not image_path:
        raise gr.Error("No image provided. Please upload an image.")

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    try:
        img, gray, binary = preprocess_image(image_path, method="adaptive")
    except Exception as e:
        raise gr.Error(f"Preprocessing failed: {e}\n{traceback.format_exc()}")

    # ── Step 2: Segment lines ─────────────────────────────────────────────────
    try:
        line_boxes = segment_lines(binary)
        line_boxes = sorted(line_boxes, key=lambda b: b[1])   # top → bottom
    except Exception as e:
        raise gr.Error(f"Line segmentation failed: {e}")

    if not line_boxes:
        vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msg = "No text lines detected. Try a clearer image."
        return vis_img, msg, msg, msg

    # ── Step 3: OCR each line with TrOCR ─────────────────────────────────────
    vis_img = img.copy()
    extracted_text_lines = []

    for (x, y, w, h) in line_boxes:

        # ── 3a: Reject lines that are too thin ────────────────────────────────
        if h < _MIN_LINE_HEIGHT:
            continue

        # Crop colour image for TrOCR input
        line_img_bgr = img[y: y + h, 0: img.shape[1]]

        # ── 3b: Reject nearly-blank / border lines using binary mask ──────────
        line_binary = binary[y: y + h, 0: binary.shape[1]]
        if not _line_has_ink(line_binary):
            continue

        # ── 3c: Reject lines that are too thin after the ink check ────────────
        if line_img_bgr.shape[0] < 4 or line_img_bgr.shape[1] < 4:
            continue

        # Draw a green bounding box on the visualisation only for kept lines
        cv2.rectangle(vis_img, (0, y), (img.shape[1], y + h), (0, 200, 0), 2)

        try:
            pil_img = Image.fromarray(cv2.cvtColor(line_img_bgr, cv2.COLOR_BGR2RGB))
            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

            generated_ids = trocr_model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True,
            )
            ocr_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # ── 3d: Post-filter: keep only outputs that look like real text ───
            if _is_real_text(ocr_output):
                extracted_text_lines.append(ocr_output.strip())
            # else: silently skip hallucinated / symbol-only outputs

        except Exception as e:
            # A single bad line must not abort the whole document
            pass   # silently skip; don't inject error strings into the output

    raw_text = "\n".join(extracted_text_lines).strip()

    if not raw_text:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        msg = "No readable text could be extracted. Try a higher-resolution image."
        return vis_img, msg, msg, msg

    # ── Step 4: Post-processing ───────────────────────────────────────────────
    try:
        clean_text = post_processor.process(raw_text)
    except Exception:
        clean_text = raw_text   # fall back to raw on postproc error


    word_count = len(clean_text.split())

    if word_count < 30:
        summary = clean_text
    else:
        try:
            summary = summarizer.summarize(clean_text)
        except Exception as e:
            summary = f"Summarization error: {e}"

    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    return vis_img, raw_text, clean_text, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="End-to-End Handwritten Document Summarizer") as demo:
    gr.Markdown("# 📝 Handwritten Document Recognition and Summarization")
    gr.Markdown(
        "Upload a handwritten document image. "
        "The system segments lines, recognizes text with TrOCR, "
        "corrects spelling, and generates an abstractive summary using T5."
    )

    with gr.Row():
        with gr.Column():
            img_input   = gr.Image(type="filepath", label="Upload Document Image")
            submit_btn  = gr.Button("Process Document", variant="primary")
            vis_output  = gr.Image(type="numpy", label="Detected Text Lines")

        with gr.Column():
            raw_output     = gr.Textbox(label="Raw Extracted Text (TrOCR output)", lines=6)
            clean_output   = gr.Textbox(label="Cleaned & Spell-Corrected Text",    lines=6)
            summary_output = gr.Textbox(label="Abstractive Summary",               lines=5)

    submit_btn.click(
        fn=process_pipeline,
        inputs=[img_input],
        outputs=[vis_output, raw_output, clean_output, summary_output],
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    server = os.environ.get("SERVER_NAME", "0.0.0.0")
    demo.launch(server_name=server, server_port=port, share=False)