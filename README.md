# Hybrid Query System

## Overview
This project explores building a **hybrid query system** that can answer natural language questions over both structured CVE data (SQL) and unstructured descriptions (vector search). The goal is to implement the core logic without relying on heavy frameworks like LangChain or LlamaIndex — instead, everything is assembled from first principles using lightweight libraries.

The system is evolving into a minimal **retrieval-augmented generation (RAG)** pipeline with the following components:

- **Structured Search (SQL):**  
  CVEs ingested into SQLite tables, allowing for filtering and retrieval by fixed fields (e.g., CVE ID, CWE, severity, publication date).
  
- **Unstructured Search (Vector Index):**  
  Natural language queries matched against CVE descriptions via vector embeddings stored in a SQLite `vec0` table.

- **Hybrid Retrieval & Routing:**  
  Future logic will allow routing queries to either/both retrieval paths depending on intent (e.g., metadata filters vs semantic search).

- **Answer Synthesis:**  
  Retrieved context will be passed to a small local LLM (TinyLlama) to generate coherent natural-language answers.

---

## Key Components

### Ingestion
- Parse NVD CVE JSON feeds.
- Flatten and normalize fields (CVE ID, description, severity, CWE, published/modified dates).
- Store them in SQLite (`cves` table).

### TF-IDF Search
- A baseline text retrieval engine built with `scikit-learn`’s `TfidfVectorizer`.
- Indexes CVE descriptions, stores the sparse matrix + vocab with `joblib`.
- Supports top-k search for fast prototyping.

### Embedding Search (FastEmbed + sqlite-vec)
- Dense embeddings generated via [FastEmbed](https://github.com/qdrant/fastembed).
- Chosen model: **MiniLM-L6-v2** (384-dim, very fast) for this PoC.
- Stored in a `vec0` virtual table inside SQLite, powered by the [`sqlite-vec`](https://github.com/asg017/sqlite-vec) extension.
- Supports efficient nearest neighbor search with `MATCH vec_f32(?)` queries.

### Routing & Reranking (Planned)
- Introduce a **router** that decides whether a query should hit SQL, TF-IDF, embeddings, or a combination.
- Reranking results by relevance score or cross-encoder embeddings.
- Generate answers by combining multiple evidence sources.

### Answer Generation
- Current prototype targets **TinyLlama 1.1B (GGUF)** as the local LLM for synthesis.
- The plan is to feed top-k retrieved contexts and produce fluent answers without external APIs.

### Structured Queries (Future Work)
- Extend the pipeline to recognize when queries involve structured filters (e.g., *“high severity CVEs published after 2023”*).
- Use an LLM to generate valid SQL queries over the CVE schema.
- Combine structured filters with semantic search for hybrid answers.

---

## Why These Choices
- **SQLite** → portable, no infra, handles both structured data and vector search with `sqlite-vec`.  
- **FastEmbed** → lightweight, ONNX-optimized, avoids transformer overhead while still providing competitive embeddings.  
- **MiniLM-L6-v2** → smallest model that’s “good enough”, prioritizing speed for prototyping.  
- **TinyLlama** → local inference, no external dependencies, keeps the whole system self-contained.  
- **No Frameworks** → forces clarity on each component: ingestion, indexing, retrieval, routing, synthesis.

---

## Roadmap
- ✅ Data ingestion into SQLite  
- ✅ TF-IDF search baseline  
- ✅ Dense embeddings via FastEmbed  
- ✅ Store embeddings in `sqlite-vec`  
- 🔄 Router logic (hybrid SQL + vector retrieval)  
- 🔄 Reranking with stronger embeddings  
- 🔄 Local LLM integration for synthesis  
- 🔮 Structured query translation with SQL generation  

---

## Usage
The system is CLI-driven. Examples:

```bash



# Build TF-IDF index
python src/tfidf.py build --db data/db/docstore.db --model models/tfidf

# Search via TF-IDF
python src/tfidf.py search --model models/tfidf --query "openssl buffer overflow"

# Generate embeddings + store in SQLite
python src/embeddings.py build --db data/db/docstore.db

# Vector search
python src/embeddings.py search --db data/db/docstore.db --query "remote code execution in apache"
```

⚡ This is an active exploration project: expect it to evolve rapidly as new components (router, reranking, answer generation) are integrated.
