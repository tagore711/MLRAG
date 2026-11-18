"""Live RAG Triad Metrics: Groundedness (Faithfulness), Context Relevance, and Latency.

Evaluates RAG execution across the industry-standard RAG Triad:
1. Faithfulness / Groundedness: Is every sentence mathematically supported by the retrieved context?
2. Context Relevance: What is the signal-to-noise ratio of the retrieved chunks to the user's question?
3. Round-trip Latency: Exact execution time in milliseconds.
"""

import re
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

EXACT_NOT_FOUND_RESPONSE = "The answer is not available in the provided documents."


def _split_into_sentences(text: str) -> List[str]:
    """Split text into individual sentences, ignoring bracketed citations."""
    # Remove citation tags like [doc.pdf, p.1] temporarily for sentence segmentation
    clean_text = re.sub(r"\[[^\]]+\]", "", text).strip()
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", clean_text)
    meaningful = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
    return meaningful if meaningful else ([clean_text] if clean_text else [])


def compute_groundedness(answer: str, context_docs: List[Document]) -> float:
    """Compute mathematical Groundedness (Faithfulness) score (0.0 to 1.0).

    Verifies whether each sentence in the generated answer is grounded in the retrieved context.
    - If the LLM truthfully returns the exact fallback refusal, groundedness is 1.0 (100% faithful).
    - If there are no context docs and an answer is claimed, groundedness is 0.0.
    - Otherwise, computes token containment and n-gram overlap between each answer sentence and context.
    """
    cleaned_ans = answer.strip()
    if not cleaned_ans or cleaned_ans == EXACT_NOT_FOUND_RESPONSE:
        return 1.0

    if not context_docs:
        return 0.0

    # Combine context into lowercased token set
    full_context_text = " ".join([d.page_content for d in context_docs]).lower()
    context_tokens = set(re.findall(r"\w+", full_context_text))

    sentences = _split_into_sentences(cleaned_ans)
    if not sentences:
        return 1.0

    grounded_sentence_count = 0
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "it", "this", "that", "with", "by", "from", "as"}

    for sent in sentences:
        sent_tokens = set(re.findall(r"\w+", sent.lower()))
        content_tokens = sent_tokens - stop_words
        if not content_tokens:
            grounded_sentence_count += 1
            continue

        overlap = len(content_tokens.intersection(context_tokens))
        overlap_ratio = overlap / len(content_tokens)

        # A sentence is considered grounded if >= 60% of its content words are verified in context,
        # or if an exact substring of 4+ consecutive words appears verbatim in context.
        has_verbatim_phrase = False
        words = sent.lower().split()
        if len(words) >= 4:
            for i in range(len(words) - 3):
                phrase = " ".join(words[i:i+4])
                if phrase in full_context_text:
                    has_verbatim_phrase = True
                    break

        if overlap_ratio >= 0.60 or has_verbatim_phrase:
            grounded_sentence_count += 1

    score = grounded_sentence_count / len(sentences)
    return round(max(0.0, min(1.0, score)), 2)


def compute_context_relevance(
    question: str,
    context_docs: List[Document],
    grade_score: Optional[float] = None,
) -> float:
    """Compute Context Relevance score (0.0 to 1.0) measuring signal vs noise.

    Args:
        question: User query string.
        context_docs: Retrieved document chunks.
        grade_score: Optional relevance score from CRAG grader (if available).

    Returns:
        Float between 0.0 and 1.0.
    """
    if not context_docs:
        return 0.0

    # Extract question keywords
    q_words = set(re.findall(r"\w+", question.lower()))
    stop_words = {"what", "is", "the", "in", "of", "and", "a", "an", "to", "for", "on", "with", "at", "by", "from", "are", "was", "were", "do", "does", "did", "tell", "me", "how", "who", "which"}
    meaningful_q_words = q_words - stop_words

    if not meaningful_q_words:
        return grade_score if grade_score is not None else 0.85

    # Measure proportion of chunks that contain core query terms
    chunks_with_hits = 0
    total_term_hits = 0

    for doc in context_docs:
        doc_words = set(re.findall(r"\w+", doc.page_content.lower()))
        hits = len(meaningful_q_words.intersection(doc_words))
        if hits > 0:
            chunks_with_hits += 1
            total_term_hits += hits

    hit_ratio = chunks_with_hits / len(context_docs)
    density_score = min(1.0, total_term_hits / (len(meaningful_q_words) * len(context_docs)))
    lexical_score = round(0.6 * hit_ratio + 0.4 * density_score, 2)

    if grade_score is not None:
        # Fuse LLM CRAG relevance score with lexical density (70% LLM, 30% lexical)
        return round(0.7 * grade_score + 0.3 * lexical_score, 2)

    return lexical_score


def compute_rag_triad(
    question: str,
    answer: str,
    context_docs: List[Document],
    latency_ms: float,
    grade_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute full live RAG Triad metrics dictionary.

    Returns:
        Dictionary with groundedness, context_relevance, latency, and badge components.
    """
    groundedness = compute_groundedness(answer, context_docs)
    context_rel = compute_context_relevance(question, context_docs, grade_score=grade_score)

    groundedness_pct = int(round(groundedness * 100))
    context_rel_pct = int(round(context_rel * 100))
    latency_int = int(round(latency_ms))

    # Determine status emojis based on industry benchmarks
    grounded_emoji = "🟢" if groundedness_pct >= 85 else ("🟡" if groundedness_pct >= 65 else "🔴")
    context_emoji = "🔵" if context_rel_pct >= 75 else ("🟡" if context_rel_pct >= 50 else "🔴")

    badge_markdown = (
        f"{grounded_emoji} **Groundedness:** `{groundedness_pct}%` &nbsp;|&nbsp; "
        f"{context_emoji} **Context Relevance:** `{context_rel_pct}%` &nbsp;|&nbsp; "
        f"⚡ **Latency:** `{latency_int}ms`"
    )

    badge_html = (
        f'<div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; margin-bottom:4px;">'
        f'<span style="background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.25); border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600;">{grounded_emoji} Grounded: {groundedness_pct}%</span>'
        f'<span style="background:rgba(59,130,246,0.12); color:#60a5fa; border:1px solid rgba(59,130,246,0.25); border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600;">{context_emoji} Context: {context_rel_pct}%</span>'
        f'<span style="background:rgba(245,158,11,0.12); color:#fbbf24; border:1px solid rgba(245,158,11,0.25); border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600;">⚡ {latency_int}ms</span>'
        f'</div>'
    )

    return {
        "groundedness": groundedness,
        "context_relevance": context_rel,
        "latency_ms": latency_ms,
        "groundedness_pct": groundedness_pct,
        "context_relevance_pct": context_rel_pct,
        "latency_int": latency_int,
        "badge_markdown": badge_markdown,
        "badge_html": badge_html,
    }

