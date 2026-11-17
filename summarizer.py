
import re
import math
from collections import Counter


def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def extract_key_takeaways(text, num_bullets=4):
    sentences = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
        if len(s.strip().split()) > 6
    ]

    if not sentences:
        return ["Document contains concise direct content."]

    stop_words = {
        "the", "and", "is", "in", "to", "of", "a", "that",
        "it", "with", "as", "for", "was", "on", "are", "by",
        "this", "be", "an", "at", "from", "or", "which",
        "were", "but", "not", "have", "has", "had", "they",
        "their", "we", "our", "you", "your", "can", "all"
    }

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    freq = Counter(w for w in words if w not in stop_words)

    scored = []

    for sentence in sentences:
        sentence_words = [
            w for w in re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
            if w not in stop_words
        ]

        if sentence_words:
            score = sum(freq[w] for w in sentence_words) / math.sqrt(len(sentence_words))
            scored.append((score, sentence))

    scored.sort(reverse=True)
    return [s[1] for s in scored[:num_bullets]]


def fallback_extractive_summary(text, num_sentences=4):
    bullets = extract_key_takeaways(text, num_sentences)
    return " ".join(bullets) if bullets else text[:500]


def generate_summary(text, summary_mode="detailed",
                     max_tokens=100, min_tokens=20):

    text = clean_text(text)

    if not text:
        return "No text available to summarize."

    # Keep processing lightweight for Render free plan
    if len(text.split()) > 600:
        text = " ".join(text.split()[:600])

    # Use extractive summarization; do not load T5/PyTorch
    return fallback_extractive_summary(text)


def analyze_document_text(text, summary_mode="detailed"):
    words = text.split()
    original_count = len(words)

    original_time = max(1, math.ceil(original_count / 220))

    summary = generate_summary(text, summary_mode)
    summary_count = len(summary.split())

    summary_time = max(1, math.ceil(summary_count / 220))

    reduction = max(
        0,
        min(95, round(
            (1 - summary_count / max(1, original_count)) * 100
        ))
    )

    return {
        "summary": summary,
        "takeaways": extract_key_takeaways(text, 4),
        "metrics": {
            "original_word_count": original_count,
            "summary_word_count": summary_count,
            "reduction_percentage": reduction,
            "original_reading_time_mins": original_time,
            "summary_reading_time_mins": summary_time,
            "time_saved_mins": max(0, original_time - summary_time),
            "model_used": "Lightweight Extractive Summarization"
        }
    }
