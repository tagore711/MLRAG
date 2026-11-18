"""Streamlit user interface for MLRAG: Multilingual Multi-PDF Chat App."""

import logging
import os
import time
from pathlib import Path
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS

from rag.chain import (
    EXACT_NOT_FOUND_RESPONSE,
    create_lcel_chain,
    format_context_and_cap,
    get_llm,
    stream_chain_with_backoff,
)
from rag.config import get_config
from rag.hybrid import (
    SimpleBM25,
    compare_and_retrieve,
)
from rag.metrics import compute_rag_triad
from rag.loader import load_pdf
from rag.splitter import split_documents
from rag.vectorstore import (
    clear_vectorstore,
    create_or_update_vectorstore,
    get_embeddings,
    load_vectorstore,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="MLRAG - Multilingual PDF Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Branding */
    .gradient-title {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }

    .sub-caption {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    /* Modern Feature Cards */
    .feature-box {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
        transition: all 0.25s ease;
    }
    .feature-box:hover {
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(99, 102, 241, 0.04);
        transform: translateY(-2px);
    }
    .feature-box-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .feature-box-desc {
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-active {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }
    .status-idle {
        background: rgba(148, 163, 184, 0.1);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    /* Chat bubble aesthetic */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background-color: rgba(255, 255, 255, 0.015);
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_embeddings(model_name: str):
    """Cache the embedding model in memory across Streamlit reruns."""
    return get_embeddings(model_name=model_name)


@st.cache_resource(show_spinner=False)
def get_cached_llm(model_name: str, api_key: str):
    """Reuse persistent HTTP/2 connection pool for Groq LLM across chat turns."""
    return get_llm(api_key=api_key, model_name=model_name)


def init_session_state():
    """Initialize necessary Streamlit session state variables with lazy loading."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "bm25" not in st.session_state:
        st.session_state.bm25 = None

    config = get_config()

    # Lazy-load vector store: only load embeddings if an index already exists on disk
    if st.session_state.get("vectorstore") is None:
        index_dir = Path(config.faiss_index_path)
        if (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
            try:
                embeddings = get_cached_embeddings(config.embed_model)
                loaded_vs = load_vectorstore(embeddings=embeddings, index_path=config.faiss_index_path)
                st.session_state.vectorstore = loaded_vs
                if loaded_vs and hasattr(loaded_vs, "docstore") and hasattr(loaded_vs.docstore, "_dict"):
                    st.session_state.bm25 = SimpleBM25(list(loaded_vs.docstore._dict.values()))
            except Exception as e:
                logger.error(f"Error auto-loading vectorstore: {e}")
                st.session_state.vectorstore = None
        else:
            st.session_state.vectorstore = None


def get_index_details(vs: Optional[FAISS] = None) -> dict:
    """Inspect active FAISS vectorstore or persisted files on disk and return structured metadata."""
    config = get_config()
    index_dir = Path(config.faiss_index_path)

    # 1. Check in-memory instance
    if vs is not None:
        try:
            total_chunks = vs.index.ntotal if hasattr(vs, "index") else 0
            sources = set()
            if hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
                for doc in vs.docstore._dict.values():
                    if hasattr(doc, "metadata"):
                        sources.add(doc.metadata.get("source", "document"))
            if total_chunks > 0:
                return {
                    "is_indexed": True,
                    "total_chunks": total_chunks,
                    "sources": sorted(list(sources)),
                }
        except Exception as e:
            logger.warning(f"Error reading in-memory vectorstore: {e}")

    # 2. Check persisted FAISS files on disk
    if (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
        try:
            import pickle
            with open(index_dir / "index.pkl", "rb") as f:
                docstore, index_to_docstore_id = pickle.load(f)
            sources = set()
            for doc_id in index_to_docstore_id.values():
                doc = docstore.search(doc_id)
                if hasattr(doc, "metadata"):
                    sources.add(doc.metadata.get("source", "document"))
            total = len(index_to_docstore_id)
            if total > 0:
                return {
                    "is_indexed": True,
                    "total_chunks": total,
                    "sources": sorted(list(sources)),
                }
        except Exception as e:
            logger.warning(f"Error reading index on disk: {e}")

    return {"is_indexed": False, "total_chunks": 0, "sources": []}


def main():
    config = get_config()
    init_session_state()

    # --- Sidebar: Document Management & Settings ---
    with st.sidebar:
        st.title("📚 MLRAG Manager")
        st.caption("Multilingual Multi-PDF RAG Engine")

        st.markdown("---")
        st.subheader("🔑 Groq Credentials")
        api_key_env = config.groq_api_key
        api_key_input = st.text_input(
            "Groq API Key",
            value=api_key_env,
            type="password",
            placeholder="gsk_...",
            help="Free Groq API key from https://console.groq.com/keys",
        )
        active_api_key = api_key_input.strip() if api_key_input else api_key_env

        st.markdown("---")
        st.subheader("⚙️ Model Configuration")

        # Groq Models: qwen3.8-27b provides ultra-low latency (~260ms), gpt-oss-120b provides deep reasoning
        groq_model_options = [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            config.groq_model,
        ]
        unique_groq_models = list(dict.fromkeys(groq_model_options))
        selected_groq_model = st.selectbox(
            "Groq LLM Model",
            options=unique_groq_models,
            index=0,
            help="qwen/qwen3.8-27b: Ultra-low latency (~260ms). openai/gpt-oss-120b: Deep reasoning.",
        )

        # Embedding Models
        embed_options = [
            config.embed_model,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]
        unique_embed_options = list(dict.fromkeys(embed_options))
        selected_embed_model = st.selectbox(
            "Embedding Model",
            options=unique_embed_options,
            index=0,
            help=(
                "BAAI/bge-m3: High accuracy (2.2 GB download on first run).\n"
                "paraphrase-multilingual-MiniLM-L12-v2: Fast & lightweight (470 MB)."
            ),
        )

        st.markdown("---")
        st.subheader("🔀 Retrieval Strategy")
        enable_hybrid = st.toggle(
            "Hybrid Search (BM25 + FAISS)",
            value=True,
            help="Merges sparse keyword BM25 with dense FAISS embeddings via Reciprocal Rank Fusion (RRF).",
        )
        show_triad = st.toggle(
            "📊 Live RAG Triad Badges",
            value=True,
            help="Displays Groundedness (Faithfulness), Context Relevance, and Latency for every answer.",
        )

        st.markdown("---")
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or multiple PDF documents. Scanned pages are automatically OCR-processed.",
        )

        col_proc, col_clr = st.columns(2)
        with col_proc:
            process_btn = st.button("Process PDFs", type="primary", use_container_width=True)
        with col_clr:
            clear_btn = st.button("Clear Index", type="secondary", use_container_width=True)

        # Dynamic, accurate Index Status calculation
        idx_info = get_index_details(st.session_state.vectorstore)

        if idx_info["is_indexed"]:
            source_names = ", ".join(idx_info["sources"])
            st.success(f"🟢 **Index: Active & Ready**\n\n📄 `{source_names}` ({idx_info['total_chunks']} chunks)")
        elif uploaded_files:
            st.warning(f"🟡 **{len(uploaded_files)} File(s) Staged**\n\nClick **Process PDFs** above to index.")
        else:
            st.info("⚪ **Index Status: Empty**\n\nUpload a PDF above to begin.")

        if process_btn:
            if not uploaded_files:
                st.sidebar.error("Please upload at least one PDF file first.")
            else:
                with st.spinner("Extracting per-page text & running OCR fallback if needed..."):
                    all_raw_docs = []
                    for uploaded_file in uploaded_files:
                        pdf_bytes = uploaded_file.read()
                        uploaded_file.seek(0)

                        docs = load_pdf(
                            file_source=pdf_bytes,
                            filename=uploaded_file.name,
                            ocr_languages=config.ocr_languages,
                            min_char_count=config.min_page_char_count,
                        )
                        all_raw_docs.extend(docs)

                if not all_raw_docs or not any(d.page_content.strip() for d in all_raw_docs):
                    st.sidebar.warning("No readable text could be extracted from uploaded files.")
                else:
                    with st.spinner("Chunking text with metadata preservation..."):
                        chunks = split_documents(
                            documents=all_raw_docs,
                            chunk_size=config.chunk_size,
                            chunk_overlap=config.chunk_overlap,
                        )

                    with st.spinner(f"Loading embedding model '{selected_embed_model}' (downloads once on first run)..."):
                        embeddings = get_cached_embeddings(selected_embed_model)

                    with st.spinner("Embedding chunks and updating FAISS vector store..."):
                        vs = create_or_update_vectorstore(
                            documents=chunks,
                            embeddings=embeddings,
                            index_path=config.faiss_index_path,
                        )
                        st.session_state.vectorstore = vs
                        if vs and hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
                            st.session_state.bm25 = SimpleBM25(list(vs.docstore._dict.values()))

                    st.sidebar.success(
                        f"✅ Indexed {len(all_raw_docs)} pages into {len(chunks)} searchable chunks!"
                    )
                    st.rerun()

        if clear_btn:
            success = clear_vectorstore(config.faiss_index_path)
            st.session_state.vectorstore = None
            st.session_state.bm25 = None
            st.session_state.messages = []
            if success:
                st.sidebar.info("Index and chat history cleared.")
            else:
                st.sidebar.error("Failed to delete local FAISS index files.")
            st.rerun()

    # --- Main Content: Chat Interface ---
    head_col1, head_col2 = st.columns([0.76, 0.24])
    with head_col1:
        st.markdown('<div class="gradient-title">🌐 MLRAG Document Intelligence</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-caption">'
            'Ultra-fast multilingual document assistant &nbsp;·&nbsp; Powered by <b>Groq LPUs</b> &nbsp;·&nbsp; '
            'Strict zero-hallucination grounding with exact citations'
            '</div>',
            unsafe_allow_html=True,
        )
    with head_col2:
        if idx_info["is_indexed"]:
            source_lbl = f"{len(idx_info['sources'])} PDF(s) Active"
            st.markdown(
                f'<div style="text-align: right; padding-top: 10px; margin-bottom: 8px;">'
                f'<span class="status-pill status-active">🟢 {source_lbl} ({idx_info["total_chunks"]} chunks)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif uploaded_files:
            st.markdown(
                f'<div style="text-align: right; padding-top: 10px; margin-bottom: 8px;">'
                f'<span class="status-pill" style="background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3);">🟡 {len(uploaded_files)} Staged</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align: right; padding-top: 10px; margin-bottom: 8px;">'
                '<span class="status-pill status-idle">⚪ No PDFs Indexed</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        if st.session_state.messages:
            if st.button("🧹 Clear Chat", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.rerun()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Live RAG Triad Badge
            if msg.get("triad") and show_triad and msg.get("content", "").strip() != EXACT_NOT_FOUND_RESPONSE:
                st.markdown(msg["triad"].get("badge_html", msg["triad"]["badge_markdown"]), unsafe_allow_html=True)

    # User Chat Input
    if prompt := st.chat_input("Ask any question about your uploaded documents in any language..."):
        if not active_api_key:
            st.error("Please enter a valid Groq API Key in the sidebar or define GROQ_API_KEY in .env.")
            st.stop()

        # Append and display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant turn
        with st.chat_message("assistant"):
            t_start = time.perf_counter()
            if st.session_state.vectorstore is None:
                response_text = EXACT_NOT_FOUND_RESPONSE
                sources_info = []
                benchmark_info = None
                triad_info = None
                st.markdown(response_text)
            else:
                chat_llm = get_cached_llm(selected_groq_model, active_api_key)

                # 1. Multi-Strategy Hybrid Retrieval (FAISS + BM25)
                with st.spinner("Searching documents with hybrid retrieval..."):
                    retrieval_res = compare_and_retrieve(
                        query=prompt,
                        vectorstore=st.session_state.vectorstore,
                        bm25=st.session_state.bm25,
                        reranker=None,
                        use_hybrid=enable_hybrid,
                        use_reranker=False,
                        top_k=config.top_k,
                    )
                    chosen_docs = retrieval_res["chosen"]
                    benchmark_info = retrieval_res

                    formatted_context, source_docs = format_context_and_cap(
                        docs=chosen_docs,
                        max_tokens=config.max_context_tokens,
                    )

                if not chosen_docs:
                    response_text = EXACT_NOT_FOUND_RESPONSE
                    sources_info = []
                    triad_info = None
                    st.markdown(response_text)
                else:
                    sources_info = [
                        {"metadata": doc.metadata, "content": doc.page_content}
                        for doc in source_docs
                    ]

                    # 2. Real-time token streaming with st.write_stream
                    chain = create_lcel_chain(chat_llm)
                    stream_gen = stream_chain_with_backoff(
                        chain=chain,
                        inputs={"context": formatted_context, "question": prompt},
                    )
                    response_text = st.write_stream(stream_gen)

                    # Compute round-trip latency & Live RAG Triad
                    latency_ms = (time.perf_counter() - t_start) * 1000
                    triad_info = compute_rag_triad(
                        question=prompt,
                        answer=response_text,
                        context_docs=source_docs,
                        latency_ms=latency_ms,
                    )

                    # 3. Live RAG Triad Badge Display
                    if show_triad and response_text.strip() != EXACT_NOT_FOUND_RESPONSE:
                        st.markdown(triad_info.get("badge_html", triad_info["badge_markdown"]), unsafe_allow_html=True)

            # Persist assistant message to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources_info,
                "benchmark": benchmark_info,
                "triad": triad_info,
                "id": len(st.session_state.messages),
            })


if __name__ == "__main__":
    main()
