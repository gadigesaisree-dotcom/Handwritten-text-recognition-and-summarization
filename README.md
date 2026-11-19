# 📝 Handwritten Text Recognition and Summarization

> An end-to-end deep learning pipeline that converts handwritten document images into machine-readable text and generates concise abstractive summaries.

---

## 📌 Overview

This project implements a complete pipeline for **Handwritten Text Recognition (HTR)** followed by **automatic text summarization**. Given a scanned or photographed handwritten document, the system:

1. **Preprocesses** the image (grayscale conversion, noise removal, binarization)
2. **Segments** the image into individual text lines
3. **Recognizes** text using Microsoft's **TrOCR** (Transformer-based OCR)
4. **Post-processes** the recognized text (normalization + spell correction)
5. **Summarizes** the extracted text using **Google's Flan-T5** model

The entire pipeline is exposed through an interactive **Gradio** web interface.

---

## 🎯 Motivation

Digitizing handwritten documents is a critical challenge in many domains — healthcare records, historical archives, academic notes, and legal documents. Manual transcription is time-consuming and error-prone. This project automates the process using state-of-the-art transformer models and provides a summarization layer for quick comprehension of long documents.

---

## 🧠 System Architecture

```
Input Image
    │
    ▼
┌─────────────────────────┐
│   Image Preprocessing   │  ← Grayscale, Gaussian blur, Adaptive thresholding
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Line Segmentation     │  ← Horizontal projection profiles
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Text Recognition      │  ← Microsoft TrOCR (Vision Encoder-Decoder)
│   (TrOCR)               │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Post-Processing       │  ← Text normalization + Spell correction
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Text Summarization    │  ← Google Flan-T5 (Abstractive Summarization)
└─────────────────────────┘
    │
    ▼
Gradio Web Interface (Visualization + Text + Summary)
```

---

## 🔬 Models Used

| Component | Model | Source |
|-----------|-------|--------|
| Handwritten Text Recognition | `microsoft/trocr-base-handwritten` | Hugging Face |
| Text Summarization | `google/flan-t5-base` | Hugging Face |
| Spell Correction | `pyspellchecker` | Python Library |
| Alternative HTR (CRNN) | Custom CNN + BiLSTM + CTC | Built from scratch |

### TrOCR
TrOCR is a Transformer-based OCR model using a **ViT encoder** and a **RoBERTa-based decoder**. It is pre-trained on large-scale handwritten datasets including IAM and SROIE. The model uses beam search decoding with:
- `num_beams = 4`
- `length_penalty = 2.0`
- `no_repeat_ngram_size = 3`

### CRNN (Custom, for training)
A custom **CNN-BiLSTM** architecture with CTC loss is also available for training on custom datasets:
- CNN backbone extracts visual features
- Two-layer Bidirectional LSTM captures sequential dependencies
- CTC loss handles alignment-free sequence labeling

### Flan-T5 Summarization
Uses `google/flan-t5-base` with:
- `min_length = 30`, `max_length = 130`
- `num_beams = 4`, `length_penalty = 2.0`
- Input prefix: `"summarize: <text>"`

---

## 📂 Project Structure

```
NLPproject/
├── Project/
│   ├── app.py               # Main Gradio application
│   ├── config.yaml          # Model and training configuration
│   ├── requirements.txt     # Python dependencies
│   └── read_pdfs.py         # PDF reference reader utility
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Image preprocessing & line segmentation
│   ├── htr_models.py        # CRNN and TrOCR model definitions
│   ├── dataset.py           # IAM dataset loader
│   ├── train_htr.py         # Training scripts for CRNN and TrOCR
│   ├── postprocessing.py    # Text normalization & spell correction
│   ├── summarization.py     # T5-based summarization module
│   └── metrics.py           # CER, WER, and ROUGE evaluation metrics
│
├── References/              # Research papers and project report
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended) or CPU

### 1. Clone the repository

```bash
git clone https://github.com/TagoreC/Handwritten-to-text-recognition-and-summarization.git
cd Handwritten-to-text-recognition-and-summarization
```

### 2. Create a virtual environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r Project/requirements.txt
```

> **Note:** If you have a GPU, install the CUDA version of PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🚀 Running the Application

```bash
python Project/app.py
```

The Gradio interface will launch at `http://127.0.0.1:7860`. A public shareable link is also generated.

### What the Interface Shows:
| Output | Description |
|--------|-------------|
| **Detected Text Lines** | Original image with green bounding boxes around detected lines |
| **Raw Extracted Text** | Direct TrOCR output per line |
| **Cleaned & Spell-Corrected Text** | Post-processed version |
| **Abstractive Summary** | T5-generated summary (if text > 30 words) |

---

## 🔧 Configuration

Edit `Project/config.yaml` to customize model settings:

```yaml
models:
  trocr:
    pretrained_model: "microsoft/trocr-base-handwritten"
  summarization:
    pretrained_model: "google/flan-t5-base"

training:
  batch_size: 8
  epochs: 10
  learning_rate: 0.0001
  device: "cuda"  # or "cpu"
```

---

## 🏋️ Training (Optional)

If you want to fine-tune the CRNN model on your own dataset:

```bash
# Train CRNN model
python -m src.train_htr

# Train TrOCR model
python -m src.train_htr trocr
```

Prepare your dataset as a CSV with columns: `image_path`, `text`.

---

## 📊 Evaluation Metrics

The system is evaluated using:

| Metric | Description | Tool |
|--------|-------------|------|
| **CER** | Character Error Rate | `jiwer` |
| **WER** | Word Error Rate | `jiwer` |
| **ROUGE-1** | Unigram overlap | `rouge-score` |
| **ROUGE-2** | Bigram overlap | `rouge-score` |
| **ROUGE-L** | Longest Common Subsequence | `rouge-score` |

To evaluate the summarization module:
```bash
python -m src.summarization
```

---

## 📦 Dependencies

```
torch>=2.0.0
torchvision
transformers>=4.30.0
opencv-python>=4.8.0
numpy
pandas
gradio>=4.0.0
evaluate
jiwer
rouge-score
pyspellchecker
Pillow
sentencepiece
pyyaml
```

---

## 🌐 Deployment

This application is deployed on **Render** as a web service.

### Deploy on Render:
1. Fork this repository
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your GitHub repository
4. Set the following:
   - **Build Command:** `pip install -r Project/requirements.txt`
   - **Start Command:** `python Project/app.py`
   - **Environment:** Python 3
model link: https://handwritten-text-recognition-ka67.onrender.com
> **Note:** The first load may take a few minutes as Hugging Face models are downloaded automatically.

---

## 🔁 Pipeline Details

### Image Preprocessing (`src/preprocessing.py`)
- Converts color image → grayscale
- Applies Gaussian blur (5×5 kernel) for noise reduction
- **Adaptive Gaussian Thresholding** for binarization (handles uneven lighting/shadows)
- Line segmentation via **horizontal projection profiles**

### Text Recognition (`src/htr_models.py`, `Project/app.py`)
- Each line crop → RGB PIL Image → TrOCR pixel values → beam search decoding
- Post-filter: only lines with ≥2 alphabetic words (≥30% alpha ratio) are retained
- Blank/border lines rejected by ink ratio check (`< 0.5%` ink pixels → skip)

### Post-Processing (`src/postprocessing.py`)
- Regex-based normalization: strips invalid characters, collapses whitespace
- Conservative spell correction using `pyspellchecker`:
  - Skips proper nouns (capitalized words)
  - Skips numbers and short/long tokens
  - Only corrects known misspellings with sufficient frequency

### Summarization (`src/summarization.py`)
- Inputs cleaned text with prefix `"summarize: "`
- Tokenizes with max 512 tokens (T5 constraint)
- Generates summary using beam search
- Falls back to returning the cleaned text if input < 30 words

---

## 📸 Screenshots

| Feature | Preview |
|---------|---------|
| Gradio UI with uploaded document | Input image with line detection boxes |
| Extracted and corrected text | Clean readable text output |
| Generated summary | Concise abstractive summary |

---

## 📚 References

1. Li, M., et al. (2021). **TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models.** arXiv:2109.10282.
2. Raffel, C., et al. (2020). **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5).** JMLR.
3. Marti, U. V., & Bunke, H. (2002). **The IAM-database: an English sentence database for offline handwriting recognition.** IJDAR.
4. Shi, B., et al. (2017). **An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN).** IEEE TPAMI.
5. Chung, J., et al. (2014). **Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.** arXiv:1412.3555.

---

## 👤 Author

**Tagore C**  
Final Year Project — Natural Language Processing  
GitHub: [@TagoreC](https://github.com/TagoreC)

---

## 📄 License

This project is for academic purposes. All pre-trained models are subject to their respective licenses (MIT / Apache 2.0) from Hugging Face.
