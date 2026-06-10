# Apex Global — Agentic AI Research Assistant
## RAG Pipeline Demo | Unique AI Case Study

A working Python implementation of the secure RAG architecture proposed in the
Unique AI Product Expert case study. Demonstrates document ingestion, vector
retrieval, citation-grounded generation, and immutable audit logging.

---

## Architecture

```
Financial Docs (PDF/TXT)
        │
        ▼
┌─────────────────┐
│  Doc Ingestion  │  PDF page extraction, type detection
│  & Chunking     │  (broker_note / cb_minutes / transcript / filing)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TF-IDF Vector  │  Cosine similarity retrieval
│  Store          │  (swap → FinBERT embeddings in prod)
└────────┬────────┘
         │  top-K chunks
         ▼
┌─────────────────┐
│  Prompt Builder │  Injects source context + citation rules
│                 │  Enforces: no hallucination, source-only answers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Claude API     │  claude-sonnet-4-20250514
│  (private VPC   │  Set ANTHROPIC_API_KEY to enable
│   in prod)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audit Log      │  Immutable JSONL — every query, source,
│  (JSONL)        │  relevance score, and response logged
└─────────────────┘
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install anthropic pypdf scikit-learn numpy
```

### 2. Run demo (uses built-in sample docs — no API key needed)
```bash
python rag_pipeline.py --demo
```

### 3. Enable live Claude responses
```bash
export ANTHROPIC_API_KEY=your_key_here
python rag_pipeline.py --demo
```

### 4. Ingest your own documents
```bash
python rag_pipeline.py --ingest ./your_docs_folder/
```
Supports: `.pdf`, `.txt`

### 5. Query the knowledge base
```bash
# Single query
python rag_pipeline.py --query "What did the Fed say about inflation expectations?"

# Interactive mode
python rag_pipeline.py
```

### 6. View audit log
```bash
python rag_pipeline.py --audit
```

---

## Sample Documents (auto-created with --demo)

| File | Type | Content |
|------|------|---------|
| `goldman_broker_note_NVDA.txt` | broker_note | Goldman NVDA Buy rating, $1,200 PT |
| `fomc_minutes_nov2024.txt` | cb_minutes | Fed Nov 2024 rate cut decision |
| `expert_call_transcript_semis.txt` | transcript | Semiconductor supply chain expert call |

---

## Key Design Decisions (maps to case study)

### Why TF-IDF here, not dense embeddings?
TF-IDF runs with zero external dependencies — ideal for demo. In production at
Apex, replace with `text-embedding-3-large` or a FinBERT-derived model deployed
inside Apex's private VPC. Financial terminology (ticker symbols, rate terminology)
benefits from domain-adapted embeddings.

### Hallucination control
The prompt explicitly instructs Claude to:
- Only reference provided source documents
- Use `[SOURCE N]` citations on every claim
- Say "I don't know" rather than extrapolate
- Flag uncertainty explicitly

### Audit log (CRO requirement)
Every query logs:
- Timestamp (UTC)
- Full query text
- Retrieved chunks with source file, page, doc type, and relevance score
- Model used
- Full response text

This satisfies SR 11-7 model risk governance requirements.

### Document type detection
Automatically classifies ingested documents as:
`broker_note` | `cb_minutes` | `transcript` | `filing` | `unknown`

Used for metadata filtering and retrieval context.

---

## Production Upgrade Path

| Component | Demo | Production |
|-----------|------|------------|
| Embeddings | TF-IDF (sklearn) | FinBERT / text-embedding-3-large |
| Vector DB | In-memory dict | Pinecone / Weaviate (private VPC) |
| LLM | Claude via public API | Azure OpenAI / AWS Bedrock private endpoint |
| Chunking | Word-based | Semantic (spaCy sentence boundaries) |
| Auth | None | SSO + role-based access |
| Audit log | Local JSONL | Immutable cloud log (S3 + CloudTrail) |

---

## File Structure
```
apex_rag/
├── rag_pipeline.py      # Main pipeline
├── vector_store.json    # Persisted chunk index (auto-generated)
├── audit_log.jsonl      # Immutable query log (auto-generated)
├── sample_docs/         # Demo documents (auto-generated with --demo)
│   ├── goldman_broker_note_NVDA.txt
│   ├── fomc_minutes_nov2024.txt
│   └── expert_call_transcript_semis.txt
└── README.md
```
