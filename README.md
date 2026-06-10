# Apex Global — Agentic AI Research Assistant
## Unique AI Case Study | Python Toolkit

A working Python implementation of the secure AI architecture proposed for Apex Global,
a $15B AUM multi-strategy hedge fund. Built as part of the Unique AI Product Expert case study.

Two tools that work together to solve the core problem:
**analysts spending 60%+ of their day manually reading financial documents.**

---

## Tools

### 1. `rag_pipeline.py` — Document Q&A Engine
Ingest financial documents and ask natural language questions across them.
Every answer is grounded in source documents with citations — no hallucination.

### 2. `delta_detector.py` — Document Delta Detector
Compare two versions of a financial document and surface what changed,
how the tone shifted, and which changes are alpha signals worth acting on.

---

## Architecture

```
Financial Docs (PDF / TXT)
        │
        ▼
┌─────────────────────┐
│  Doc Ingestion      │  PDF extraction, auto doc-type detection
│  & Chunking         │  (broker_note / cb_minutes / transcript / filing)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌─────────────────────┐
│  RAG Pipeline       │     │  Delta Detector      │
│                     │     │                      │
│  TF-IDF vector      │     │  Sentence diff       │
│  store + cosine     │     │  Tone shift scoring  │
│  similarity         │     │  Alpha signal flags  │
│  retrieval          │     │  HTML report         │
└────────┬────────────┘     └──────────┬───────────┘
         │                             │
         └──────────┬──────────────────┘
                    ▼
         ┌─────────────────────┐
         │  Claude API         │  Synthesis + cited answers
         │  (private VPC       │  Set ANTHROPIC_API_KEY to enable
         │   in production)    │
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Audit Log (JSONL)  │  Every query, source, and output logged
         └─────────────────────┘
```

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### RAG Pipeline — Document Q&A

```bash
# Demo mode (uses built-in sample docs — no API key needed)
python3 rag_pipeline.py --demo

# Ingest your own documents
python3 rag_pipeline.py --ingest ./your_docs/

# Single query
python3 rag_pipeline.py --query "What did the Fed say about inflation?"

# Interactive Q&A mode
python3 rag_pipeline.py

# View audit log
python3 rag_pipeline.py --audit
```

### Delta Detector — Document Change Analysis

```bash
# Demo mode — compares Fed Sep vs Nov 2024 minutes
python3 delta_detector.py --demo

# Open HTML report in browser automatically
python3 delta_detector.py --demo --serve

# Compare your own documents (auto-detects type)
python3 delta_detector.py fed_sep.pdf fed_nov.pdf --serve

# Specify document type explicitly
python3 delta_detector.py broker_v1.txt broker_v2.txt --type broker_note
python3 delta_detector.py 10k_2023.pdf 10k_2024.pdf --type filing
python3 delta_detector.py fomc_oct.txt fomc_nov.txt --type cb_minutes
```

### Enable live Claude responses
```bash
export ANTHROPIC_API_KEY=your_key_here
python3 rag_pipeline.py --demo
python3 delta_detector.py --demo
```

---

## Sample Output — Delta Detector (Fed Minutes Demo)

```
🔍 Detected document type: CB Minutes
⚙️  Computing sentence-level diff...
   ✅ 0 added · 0 removed · 9 changed · 2 alpha signals
📊 Analysing tone shift...
   ⬆ More dovish language (+3 signals)

⚡ ALPHA SIGNALS (2):
  • One member dissented, preferring to hold rates steady
    citing lingering upside inflation risks.
  • Several members noted they were no longer pre-committing
    to any particular pace of further adjustments.
```

---

## Supported Document Types

| Type | Auto-detected from | Tone signals tracked |
|------|--------------------|----------------------|
| `cb_minutes` | FOMC, ECB, central bank keywords | Hawkish / Dovish |
| `filing` | 10-K, 10-Q, SEC, risk factors | Risk increase / decrease |
| `broker_note` | Price target, analyst, rating keywords | Bullish / Bearish |

---

## Key Design Decisions

### Why RAG, not fine-tuning?
Fine-tuning requires large labelled datasets and retraining cycles.
RAG gives live, verifiable answers grounded in Apex's actual documents —
critical for hallucination control. The model can only reference what
is in the knowledge base.

### Hallucination control
The prompt instructs Claude to:
- Only reference provided source documents
- Use `[SOURCE N]` citations on every claim
- Explicitly say "I don't know" rather than extrapolate
- Flag uncertainty in the response

### Audit log (CRO requirement)
Every query logs: timestamp, full query, retrieved chunks with source file,
page, doc type, relevance score, model used, and full response.
Satisfies SR 11-7 model risk governance requirements.

### Private deployment (data security)
In production, all components run inside the client's private VPC.
No data leaves their environment. Compatible with Azure OpenAI
and AWS Bedrock private endpoints.

---

## Production Upgrade Path

| Component | This Demo | Production |
|-----------|-----------|------------|
| Embeddings | TF-IDF (sklearn) | FinBERT / text-embedding-3-large |
| Vector DB | In-memory | Pinecone / Weaviate (private VPC) |
| LLM | Claude public API | Azure OpenAI / AWS Bedrock |
| Chunking | Word-based | Semantic (sentence boundaries) |
| Auth | None | SSO + role-based access |
| Audit log | Local JSONL | Immutable cloud log (S3 + CloudTrail) |

---

## File Structure
```
apex-rag-pipeline/
├── rag_pipeline.py       # Document ingestion + Q&A engine
├── delta_detector.py     # Document change analysis + tone shift
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── vector_store.json     # Auto-generated chunk index
├── audit_log.jsonl       # Auto-generated query audit trail
├── delta_report.html     # Auto-generated HTML diff report
└── sample_docs/          # Auto-generated demo documents
```

---

## Built for
Unique AI · Apex Global Design Partnership · Product Expert Case Study
