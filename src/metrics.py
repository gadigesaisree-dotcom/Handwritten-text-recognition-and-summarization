import jiwer
from rouge_score import rouge_scorer

def calculate_cer_wer(predictions, references):
    """
    Calculates Character Error Rate (CER) and Word Error Rate (WER).
    """
    cer = jiwer.cer(references, predictions)
    wer = jiwer.wer(references, predictions)
    return cer, wer

def calculate_rouge(predictions, references):
    """
    Calculates ROUGE scores for summarization.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure
        
    n = len(predictions)
    if n > 0:
        return {
            "rouge1": rouge1 / n,
            "rouge2": rouge2 / n,
            "rougeL": rougeL / n
        }
    return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

if __name__ == "__main__":
    preds = ["the quick brown fox"]
    refs = ["the quick brown fox"]
    print(calculate_cer_wer(preds, refs))
    print(calculate_rouge(preds, refs))
