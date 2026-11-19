import os
import re
import yaml
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class TextSummarizer:
    T5_MAX_INPUT_TOKENS = 512

    def __init__(self, config_path: str | None = None):
        model_name = self._resolve_model(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Summarizer] Loading T5 model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("[Summarizer] Model ready.")

    
    def summarize(
        self,
        text: str,
        min_length: int = 30,
        max_length: int = 130,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
    ) -> str:
        if not text or not text.strip():
            return ""

        clean = self._clean_for_summarization(text)

        word_count = len(clean.split())
        if word_count < 30:
            return clean   # nothing meaningful to compress

        safe_max = min(max_length, word_count)          # can't be longer than input
        safe_min = min(min_length, max(safe_max - 1, 1))

        input_text = "summarize: " + clean

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.T5_MAX_INPUT_TOKENS,
            truncation=True,
            padding=False,
        ).to(self.device)

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    min_length=safe_min,
                    max_length=safe_max,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=early_stopping,
                )
            summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return summary.strip()
        except Exception as exc:
            print(f"[Summarizer] Generation error — returning cleaned text. Reason: {exc}")
            return clean

    

    def _clean_for_summarization(self, text: str) -> str:
        text = re.sub(r"\[.*?\]", "", text)

        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)

        text = re.sub(r"\n{2,}", "\n", text)

        text = re.sub(r"[ \t]+", " ", text)

        lines = [line.strip() for line in text.splitlines()]

        lines = [ln for ln in lines if re.search(r"[A-Za-z]", ln)]

        return " ".join(lines).strip()

    def _resolve_model(self, config_path: str | None) -> str:
        default = "google/flan-t5-base"

        candidates = []
        if config_path:
            candidates.append(config_path)

        here = os.path.dirname(os.path.abspath(__file__))
        candidates += [
            os.path.join(here, "..", "Project", "config.yaml"),
            os.path.join(here, "..", "config.yaml"),
            os.path.join(here, "config.yaml"),
        ]

        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh)
                model_name = cfg["models"]["summarization"]["pretrained_model"]
                if model_name:
                    print(f"[Summarizer] Config loaded from: {path}")
                    return model_name
            except Exception:
                continue

        print(f"[Summarizer] config.yaml not found — using default: {default}")
        return default



if __name__ == "__main__":
    from src.metrics import calculate_cer_wer, calculate_rouge
    sm = TextSummarizer()

    sample = (
        "The history of Natural Language Processing generally started in the 1950s, "
        "although work can be found from earlier periods. In 1950, Alan Turing published "
        'an article titled "Computing Machinery and Intelligence" which proposed what is '
        "now called the Turing test as a criterion of intelligence. The Georgetown "
        "experiment in 1954 involved fully automatic translation of more than sixty "
        "Russian sentences into English. The authors claimed that within three or five "
        "years, machine translation would be a solved problem. However, real progress "
        "was much slower and it took decades of additional research."
    )

    print("Input:\n", sample)
    summary = sm.summarize(sample)
    print("\nSummary:\n", summary)

    # ──────────────────────────────────────────────
    # Evaluation Metrics Block
    # ──────────────────────────────────────────────
    predictions = [summary]
    references  = [sample]

    cer, wer = calculate_cer_wer(predictions, references)
    rouge     = calculate_rouge(predictions, references)

    print("\n" + "=" * 50)
    print("          EVALUATION METRICS")
    print("=" * 50)
    print(f"  ROUGE-1 : {rouge['rouge1']:.4f}")
    print(f"  ROUGE-2 : {rouge['rouge2']:.4f}")
    print(f"  ROUGE-L : {rouge['rougeL']:.4f}")
    print("-" * 50)
    print(f"  CER     : {cer:.4f}")
    print(f"  WER     : {wer:.4f}")
    print("=" * 50)