"""
Evaluation metrics for medical report generation.

Computes:
  - BLEU (1, 2, 3, 4) via nltk
  - ROUGE (1, 2, L) via rouge-score
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def ensure_nltk_data():
    """Download required NLTK data if not already present."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def compute_bleu(reference: str, hypothesis: str) -> dict:
    """
    Compute BLEU-1 through BLEU-4 for a single reference-hypothesis pair.

    Args:
        reference:  Ground truth report text.
        hypothesis: Generated report text.

    Returns:
        Dictionary with keys 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4'.
    """
    ensure_nltk_data()

    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())

    # Handle edge cases
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    smoother = SmoothingFunction().method1

    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        try:
            score = sentence_bleu(
                [ref_tokens], hyp_tokens,
                weights=weights,
                smoothing_function=smoother,
            )
        except Exception:
            score = 0.0
        scores[f"bleu_{n}"] = round(score, 6)

    return scores


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L for a single reference-hypothesis pair.

    Args:
        reference:  Ground truth report text.
        hypothesis: Generated report text.

    Returns:
        Dictionary with keys 'rouge_1', 'rouge_2', 'rouge_l' (F1 scores).
    """
    if not reference.strip() or not hypothesis.strip():
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    results = scorer.score(reference, hypothesis)

    return {
        "rouge_1": round(results["rouge1"].fmeasure, 6),
        "rouge_2": round(results["rouge2"].fmeasure, 6),
        "rouge_l": round(results["rougeL"].fmeasure, 6),
    }


def compute_all_metrics(reference: str, hypothesis: str) -> dict:
    """
    Compute all metrics (BLEU 1-4 + ROUGE 1, 2, L) for a single pair.

    Args:
        reference:  Ground truth report text.
        hypothesis: Generated report text.

    Returns:
        Dictionary with all metric scores.
    """
    bleu = compute_bleu(reference, hypothesis)
    rouge = compute_rouge(reference, hypothesis)
    return {**bleu, **rouge}


def aggregate_metrics(all_metrics: list) -> dict:
    """
    Compute average metrics across a list of per-sample metric dictionaries.

    Args:
        all_metrics: List of dicts, each from compute_all_metrics().

    Returns:
        Dictionary with averaged metric values.
    """
    if not all_metrics:
        return {}

    keys = all_metrics[0].keys()
    aggregated = {}
    for key in keys:
        values = [m[key] for m in all_metrics]
        aggregated[key] = round(sum(values) / len(values), 6)

    return aggregated
