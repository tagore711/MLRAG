"""Multilingual Evaluation Script for MLRAG.

Evaluates cross-lingual retrieval accuracy across English, Hindi, Telugu, French, German, and Spanish,
reporting Recall@1, Recall@3, and Recall@5.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Any
from langchain_core.documents import Document

from rag.hybrid import SimpleBM25, reciprocal_rank_fusion
from rag.vectorstore import FAISS


# Golden multilingual corpus of documents
GOLDEN_CORPUS = [
    Document(
        page_content="Transformers use self-attention mechanisms to compute representations of input sequences in parallel.",
        metadata={"source": "attention_paper.pdf", "page": 1, "id": "doc_attention"},
    ),
    Document(
        page_content="Gradient descent optimizes loss functions by iteratively taking steps proportional to the negative gradient.",
        metadata={"source": "optimization_guide.pdf", "page": 2, "id": "doc_optimization"},
    ),
    Document(
        page_content="Vector embeddings map words and semantic concepts into dense high-dimensional geometrical spaces.",
        metadata={"source": "embeddings_overview.pdf", "page": 5, "id": "doc_embeddings"},
    ),
    Document(
        page_content="Convolutional Neural Networks excel in visual computer vision tasks through localized kernel feature maps.",
        metadata={"source": "cv_architectures.pdf", "page": 4, "id": "doc_cnn"},
    ),
    Document(
        page_content="Recurrent neural networks process sequential data with hidden states maintaining temporal memory over time.",
        metadata={"source": "sequence_models.pdf", "page": 7, "id": "doc_rnn"},
    ),
]

# Golden Q&A evaluation benchmark across 6 languages
GOLDEN_DATASET = [
    # English
    {
        "language": "English",
        "query": "How do transformers compute sequence representations?",
        "target_id": "doc_attention",
    },
    {
        "language": "English",
        "query": "What technique maps semantic concepts into high dimensional geometric spaces?",
        "target_id": "doc_embeddings",
    },
    # Hindi
    {
        "language": "Hindi",
        "query": "ट्रांसफॉर्मर समानांतर में अनुक्रम निरूपण की गणना कैसे करते हैं?",
        "target_id": "doc_attention",
    },
    {
        "language": "Hindi",
        "query": "वेक्टर एम्बेडिंग और उच्च-आयामी ज्यामितीय स्थान क्या हैं?",
        "target_id": "doc_embeddings",
    },
    # Telugu
    {
        "language": "Telugu",
        "query": "ట్రాన్స్‌ఫార్మర్‌లు సీక్వెన్స్ ప్రాతినిధ్యాలను ఎలా లెక్కిస్తాయి?",
        "target_id": "doc_attention",
    },
    {
        "language": "Telugu",
        "query": "వెక్టర్ ఎంబెడ్డింగ్‌లు మరియు అధిక డైమెన్షనల్ స్పేస్ అంటే ఏమిటి?",
        "target_id": "doc_embeddings",
    },
    # French
    {
        "language": "French",
        "query": "Comment les transformateurs calculent-ils les représentations en parallèle?",
        "target_id": "doc_attention",
    },
    {
        "language": "French",
        "query": "Comment la descente de gradient optimise-t-elle les fonctions de coût?",
        "target_id": "doc_optimization",
    },
    # German
    {
        "language": "German",
        "query": "Wie berechnen Transformer Sequenzdarstellungen parallel?",
        "target_id": "doc_attention",
    },
    {
        "language": "German",
        "query": "Wie optimiert der Gradientenabstieg Verlustfunktionen?",
        "target_id": "doc_optimization",
    },
    # Spanish
    {
        "language": "Spanish",
        "query": "¿Cómo calculan los transformadores las representaciones en paralelo?",
        "target_id": "doc_attention",
    },
    {
        "language": "Spanish",
        "query": "¿Qué son los embeddings vectoriales en espacios geométricos densos?",
        "target_id": "doc_embeddings",
    },
]


def evaluate_retrieval(embeddings=None) -> Dict[str, Dict[str, float]]:
    """Compute Recall@1, Recall@3, and Recall@5 per language using vector and/or hybrid retrieval."""
    bm25 = SimpleBM25(GOLDEN_CORPUS)
    vs = None

    if embeddings is not None:
        try:
            vs = FAISS.from_documents(GOLDEN_CORPUS, embeddings)
        except Exception as e:
            print(f"Note: Vector store initialization failed ({e}), using lexical BM25 benchmark.")

    results_by_lang: Dict[str, List[Dict[str, bool]]] = {}

    for item in GOLDEN_DATASET:
        lang = item["language"]
        query = item["query"]
        target = item["target_id"]

        if vs is not None:
            dense_docs = vs.similarity_search(query, k=5)
            sparse_docs = bm25.search(query, top_k=5)
            retrieved = reciprocal_rank_fusion([dense_docs, sparse_docs], top_n=5)
        else:
            retrieved = bm25.search(query, top_k=5)

        retrieved_ids = [doc.metadata.get("id") for doc in retrieved]

        r1 = target in retrieved_ids[:1]
        r3 = target in retrieved_ids[:3]
        r5 = target in retrieved_ids[:5]

        if lang not in results_by_lang:
            results_by_lang[lang] = []

        results_by_lang[lang].append({"r1": r1, "r3": r3, "r5": r5})

    summary: Dict[str, Dict[str, float]] = {}
    for lang, scores in results_by_lang.items():
        total = len(scores)
        recall_1 = sum(1 for s in scores if s["r1"]) / total
        recall_3 = sum(1 for s in scores if s["r3"]) / total
        recall_5 = sum(1 for s in scores if s["r5"]) / total

        summary[lang] = {
            "Recall@1": recall_1,
            "Recall@3": recall_3,
            "Recall@5": recall_5,
            "Total": total,
        }

    return summary


def print_report(summary: Dict[str, Dict[str, float]]):
    """Format evaluation results into a clean markdown table."""
    print("\n" + "=" * 65)
    print("       MLRAG MULTILINGUAL RETRIEVAL EVALUATION REPORT")
    print("=" * 65)
    print(f"{'Language':<12} | {'Queries':<7} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10}")
    print("-" * 65)

    all_r1, all_r3, all_r5, all_count = [], [], [], 0
    for lang, m in summary.items():
        r1_pct = f"{m['Recall@1'] * 100:.1f}%"
        r3_pct = f"{m['Recall@3'] * 100:.1f}%"
        r5_pct = f"{m['Recall@5'] * 100:.1f}%"
        count = int(m["Total"])
        print(f"{lang:<12} | {count:<7} | {r1_pct:<10} | {r3_pct:<10} | {r5_pct:<10}")

        all_r1.append(m["Recall@1"])
        all_r3.append(m["Recall@3"])
        all_r5.append(m["Recall@5"])
        all_count += count

    print("-" * 65)
    avg_r1 = f"{sum(all_r1) / len(all_r1) * 100:.1f}%"
    avg_r3 = f"{sum(all_r3) / len(all_r3) * 100:.1f}%"
    avg_r5 = f"{sum(all_r5) / len(all_r5) * 100:.1f}%"
    print(f"{'OVERALL':<12} | {all_count:<7} | {avg_r1:<10} | {avg_r3:<10} | {avg_r5:<10}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    print("Running Multilingual Retrieval Evaluation...")
    summary = evaluate_retrieval(embeddings=None)
    print_report(summary)
