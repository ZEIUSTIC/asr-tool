"""Word Error Rate (WER) evaluation."""
import re


def normalize(text: str) -> list:
    """Lowercase, strip punctuation, split into words."""
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = (substitutions + deletions + insertions) / reference words."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("Reference text is empty")
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(ref)
