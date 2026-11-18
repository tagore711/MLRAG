# MLRAG: Multilingual Multi-PDF RAG Chat Application

MLRAG is a production-ready, multilingual Retrieval-Augmented Generation (RAG) chat application built entirely with free, open-source, and permissive tier components.

It allows users to upload multiple PDF documents (including scanned documents via automatic OCR fallback), index them using normalized dense multilingual embeddings, and interact with them in natural language. Queries and documents can be in completely different languages—a user can ask a question in Hindi, Telugu, French, German, or Spanish, and accurately retrieve answers from English documents (or vice versa), complete with exact page citations.

---

## 🚀 Key Features

- **Free-Tier Tech Stack**:
  - **LLM**: Groq Free Tier (`qwen/qwen3.8-27b` for ultra-low ~260ms latency or `openai/gpt-oss-120b` for deep reasoning).
  - **Embeddings**: `BAAI/bge-m3` via `langchain-huggingface` (1024-dim, normalized for cosine similarity) or lightweight `paraphrase-multilingual-MiniLM-L12-v2` (470MB).
  - **Vector Store**: FAISS (`faiss-cpu`) with automatic persistence to `./faiss_index`.
  - **Sparse Keyword Search**: Zero-dependency in-memory BM25 with Reciprocal Rank Fusion (RRF).
  - **PDF Parsing & OCR Fallback**: `pypdf` with fallback to `pdf2image` + `pytesseract` for scanned documents.
  - **Orchestration**: Modern LangChain Expression Language (LCEL) with real-time token streaming (`st.write_stream`).
- **📊 Live RAG Triad Metrics (Hallucination & Quality Meter)**:
  - **Groundedness (Faithfulness)**: Mathematical sentence-level verification against retrieved chunks.
  - **Context Relevance**: Quantitative signal-to-noise ratio in retrieved context.
  - **Latency**: Real-time round-trip latency in milliseconds (`⚡ Latency: ~260ms`).
- **Cross-Lingual Retrieval & Generation**:
  - Cross-lingual search across 100+ languages.
  - Strictly answers in the **exact same language** as the user's question.
  - Precise page-level citations: `[filename, p.N]`.
- **Hallucination & Injection Defense**:
  - If context is insufficient, replies strictly: `"The answer is not available in the provided documents."`
  - Treats retrieved document chunks as untrusted data, ignoring embedded instructions or role-reversals.
  - Caps prompt context to ~3000 tokens to ensure stability within Groq rate limits.
- **Enterprise-Grade Resilience**:
  - Exponential backoff retry handler for Groq HTTP 429 (Rate Limits) and network timeouts.
  - Safe handling of corrupt or empty PDFs without crashing.
  - Persistent index across app reruns and restarts with a one-click "Clear Index" button.

---

## 📁 Project Structure

```
MLRAG/
├── .env.example              # Environment variables template
├── .gitignore                # Excludes .env, faiss_index, cache, etc.
├── README.md                 # Setup and architecture documentation
├── requirements.txt          # Python package dependencies
├── app.py                    # Streamlit frontend with streaming & live triad badges
├── rag/
│   ├── __init__.py
│   ├── config.py             # AppConfig dataclass and environment loader
│   ├── loader.py             # Per-page PDF extraction with OCR fallback
│   ├── splitter.py           # Recursive character splitter preserving metadata
│   ├── vectorstore.py        # FAISS indexing, persistence & management
│   ├── hybrid.py             # BM25 lexical search & Reciprocal Rank Fusion
│   ├── metrics.py            # Live RAG Triad: Groundedness, Context Relevance, Latency
│   └── chain.py              # LCEL prompt, Groq backoff retry & streaming engine
├── eval/
│   └── eval_multilingual.py  # Recall@K evaluation across 6 languages
└── tests/
    ├── test_loader.py        # Tests for per-page parsing & OCR fallback trigger
    ├── test_splitter.py      # Tests for chunking & metadata preservation
    ├── test_vectorstore.py   # Tests for FAISS persistence and reloading
    ├── test_hybrid.py        # Tests for BM25 and RRF fusion
    ├── test_metrics.py       # Tests for live RAG Triad scores
    └── test_chain.py         # Tests for prompt capping, mock LLM & 429 retry
```

---

## 🔑 How to Get a Free Groq API Key

1. Go to [Groq Console](https://console.groq.com/keys).
2. Sign in or create a free account (no credit card required).
3. Navigate to **API Keys** and click **Create API Key**.
4. Copy your key (starts with `gsk_...`).
5. Paste it in your `.env` file or enter it directly into the Streamlit sidebar.

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/MLRAG.git
cd MLRAG
```

### 2. Create and activate a virtual environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Install System OCR Dependencies
For scanned PDFs, MLRAG falls back to Tesseract OCR and Poppler:
- **Ubuntu/Debian**:
  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-spa poppler-utils
  ```
- **macOS**:
  ```bash
  brew install tesseract poppler
  ```
- **Windows**:
  - Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/).
  - Add their `bin` directories to your system `PATH`.
*(Note: If Tesseract or Poppler are not installed, MLRAG gracefully extracts standard digital PDF text without crashing).*

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` with your settings:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=BAAI/bge-m3
OCR_LANGUAGES=eng+hin+tel+fra+deu+spa
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=5
MAX_CONTEXT_TOKENS=3000
FAISS_INDEX_PATH=./faiss_index
```

---

## 🖥️ Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

1. Upload one or multiple PDF documents in the sidebar.
2. Click **⚡ Process PDFs** to parse, chunk, embed, and index.
3. Type your question in any language in the chat input.
4. Expand **📚 View Retrieved Sources & Citations** under each answer to inspect exact source snippets and page numbers.
5. Use **🗑️ Clear Index** to reset the index and chat history at any time.

---

## 🌐 Supported Languages

MLRAG leverages `BAAI/bge-m3`, one of the most capable open-source multilingual embedding models supporting 100+ languages, and `llama-3.3-70b-versatile`:
- **Supported OCR Languages**: English (`eng`), Hindi (`hin`), Telugu (`tel`), French (`fra`), German (`deu`), Spanish (`spa`), and any additional language codes supported by Tesseract.
- **Cross-Lingual Capabilities**:
  - Query in Hindi $\rightarrow$ Answer from English document in Hindi.
  - Query in Telugu $\rightarrow$ Answer from English document in Telugu.
  - Query in French $\rightarrow$ Answer from German document in French.
  - Query in Spanish $\rightarrow$ Answer from English document in Spanish.

---

## ☁️ Deployment Notes & Resource Guidelines

### RAM Requirements:
- The default embedding model `BAAI/bge-m3` is ~2.2 GB on disk and requires approximately **3 GB to 4 GB of RAM** when loaded into memory with PyTorch CPU.
- **Streamlit Community Cloud Limitation**: Free Streamlit Community Cloud instances are capped at ~1 GB of RAM and may run out of memory (OOM) when loading `BAAI/bge-m3`.

### Recommended Deployment Options:
1. **Hugging Face Spaces (Free Tier)**:
   - Provides 16 GB RAM on CPU Basic instances, comfortably accommodating `BAAI/bge-m3`.
2. **Lightweight Embedding Model Swap (for low RAM / Streamlit Cloud)**:
   - If deploying to low-RAM environments (< 2 GB RAM), update `EMBED_MODEL` in `.env` to:
     ```ini
     EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
     ```
     This lightweight multilingual model requires only ~470 MB RAM while still supporting 50+ languages.

---

## ⚠️ Known Limitations

1. **Free Tier Groq Rate Limits**: Groq free tier enforces requests-per-minute (RPM) and tokens-per-minute (TPM) limits. MLRAG includes automatic exponential backoff retry and caps context tokens at 3000, but sustained heavy multi-user loads may still encounter rate limits.
2. **Scanned PDF Quality**: OCR accuracy is dependent on image resolution and Poppler/Tesseract language training packs installed on the host machine.
3. **Complex Tables & Multi-Column Layouts**: Standard `pypdf` linearizes text; tables with merged cells or complex multi-column documents may benefit from layout-aware parsing libraries (such as `pdfplumber` or `marker`).

---

## 🧪 Running Unit Tests

Run the full automated test suite with `pytest`:
```bash
pytest tests/ -v
```
All tests run with mocked LLM and synthetic PDFs, requiring no external API keys or GPU.

