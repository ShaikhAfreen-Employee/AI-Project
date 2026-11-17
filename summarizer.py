
import re
import math
from collections import Counter

# T5 is loaded only when needed
_summarizer_pipeline = None
_model_attempted = False


def get_summarizer():
    global _summarizer_pipeline, _model_attempted

    if _model_attempted:
        return _summarizer_pipeline

    _model_attempted = True

    try:
        import torch
        from transformers import pipeline

        # Limit CPU memory usage
        torch.set_num_threads(1)

        print("[BriefIQ] Loading T5-small...")

        _summarizer_pipeline = pipeline(
            "text2text-generation",
            model="t5-small",
            device=-1
        )

        print("[BriefIQ] T5-small loaded successfully!")

    except Exception as e:
        print(f"[BriefIQ] T5 unavailable: {e}")
        _summarizer_pipeline = None

    return _summarizer_pipeline


def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def split_into_chunks(text, max_chunk_words=200):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = len(sentence.split())

        if count + words > max_chunk_words and current:
            chunks.append(" ".join(current))
            current = [sentence]
            count = words
        else:
            current.append(sentence)
            count += words

    if current:
        chunks.append(" ".join(current))

    return chunks or [text]


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
    bullets = extract_key_takeaways(text, num_bullets=num_sentences)
    return " ".join(bullets) if bullets else text[:500]


def generate_summary(text, summary_mode="detailed",
                     max_tokens=100, min_tokens=20):

    text = clean_text(text)

    if not text:
        return "No text available to summarize."

    if summary_mode == "concise":
        max_tokens, min_tokens = 70, 15
    elif summary_mode == "executive":
        max_tokens, min_tokens = 120, 25
    else:
        max_tokens, min_tokens = 100, 20

    # Use extractive summary if text is very long
    if len(text.split()) > 600:
        text = " ".join(text.split()[:600])

    model = get_summarizer()

    if model is not None:
        try:
            chunks = split_into_chunks(text, 200)
            summaries = []

            # Process at most two chunks to reduce memory/time
            for chunk in chunks[:2]:
                result = model(
                    "summarize: " + chunk,
                    max_new_tokens=max_tokens,
                    min_new_tokens=min_tokens,
                    do_sample=False
                )
                summaries.append(result[0]["generated_text"].strip())

            if summaries:
                return " ".join(summaries)

        except Exception as e:
            print(f"[BriefIQ] T5 error: {e}")

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
            "model_used": (
                "Google T5-small (Deep Learning)"
                if _summarizer_pipeline is not None
                else "Extractive Summarization (Fallback)"
            )
        }
    }
