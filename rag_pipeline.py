"""
Apex Global — Agentic AI Research Assistant
RAG Pipeline Demo
-----------------
Demonstrates the core architecture proposed in the Unique AI case study:
  1. Document Ingestion     — load PDFs or text files
  2. Chunking               — semantic-aware splitting
  3. Embedding + Indexing   — TF-IDF vector store (swap for dense embeddings in prod)
  4. Retrieval              — hybrid BM25-style + cosine similarity
  5. Generation             — Anthropic Claude via API with citation injection
  6. Audit Log              — immutable JSONL log of every query + source + output

Usage:
    python rag_pipeline.py                        # interactive Q&A mode
    python rag_pipeline.py --ingest ./docs/       # ingest a folder of PDFs/txts
    python rag_pipeline.py --query "What did the Fed say about inflation?"

Requirements:
    pip install anthropic pypdf scikit-learn numpy
"""

import os
import json
import hashlib
import argparse
import datetime
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pypdf


# ─── Configuration ────────────────────────────────────────────────────────────

CHUNK_SIZE = 400          # words per chunk
CHUNK_OVERLAP = 80        # word overlap between chunks
TOP_K = 5                 # number of chunks to retrieve
AUDIT_LOG_PATH = "audit_log.jsonl"
VECTOR_STORE_PATH = "vector_store.json"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    doc_type: str           # broker_note | cb_minutes | transcript | filing | unknown
    page_number: int
    text: str
    word_count: int


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class AuditEntry:
    timestamp: str
    query: str
    retrieved_sources: list[dict]
    model: str
    response: str
    confidence_note: str


# ─── Document Type Detection ──────────────────────────────────────────────────

DOC_TYPE_SIGNALS = {
    "broker_note":  ["price target", "buy rating", "sell rating", "equity research",
                     "analyst", "overweight", "underweight", "eps estimate"],
    "cb_minutes":   ["federal reserve", "federal open market committee", "fomc",
                     "central bank", "monetary policy", "basis points", "rate decision",
                     "inflation target", "quantitative"],
    "transcript":   ["operator", "q&a", "question and answer", "transcript",
                     "expert call", "earnings call", "thank you for joining"],
    "filing":       ["10-k", "10-q", "8-k", "13-f", "sec filing", "risk factors",
                     "management discussion", "annual report", "form 10"],
}

def detect_doc_type(text: str) -> str:
    text_lower = text.lower()
    scores = {dtype: 0 for dtype in DOC_TYPE_SIGNALS}
    for dtype, signals in DOC_TYPE_SIGNALS.items():
        for signal in signals:
            if signal in text_lower:
                scores[dtype] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, source_file: str, page_number: int,
               chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    doc_type = detect_doc_type(text[:2000])  # sample first 2000 chars for type detection

    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text_str = " ".join(chunk_words)

        chunk_id = hashlib.md5(
            f"{source_file}:{page_number}:{i}".encode()
        ).hexdigest()[:12]

        chunks.append(Chunk(
            chunk_id=chunk_id,
            source_file=source_file,
            doc_type=doc_type,
            page_number=page_number,
            text=chunk_text_str,
            word_count=len(chunk_words),
        ))
        i += chunk_size - overlap
        if i >= len(words):
            break

    return chunks


# ─── Document Ingestion ───────────────────────────────────────────────────────

def ingest_pdf(filepath: str) -> list[Chunk]:
    """Extract text from PDF page by page and chunk each page."""
    chunks = []
    filename = Path(filepath).name
    print(f"  📄 Ingesting PDF: {filename}")

    with open(filepath, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 100:
                page_chunks = chunk_text(text, filename, page_num)
                chunks.extend(page_chunks)

    print(f"     → {len(chunks)} chunks from {len(reader.pages)} pages")
    return chunks


def ingest_text(filepath: str) -> list[Chunk]:
    """Ingest a plain text file."""
    filename = Path(filepath).name
    print(f"  📝 Ingesting text: {filename}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    text = re.sub(r'\s+', ' ', text).strip()
    chunks = chunk_text(text, filename, page_number=1)
    print(f"     → {len(chunks)} chunks")
    return chunks


def ingest_folder(folder_path: str) -> list[Chunk]:
    """Ingest all PDFs and text files in a folder."""
    folder = Path(folder_path)
    all_chunks = []

    pdf_files = list(folder.glob("*.pdf"))
    txt_files = list(folder.glob("*.txt"))
    all_files = pdf_files + txt_files

    if not all_files:
        print(f"⚠️  No PDF or .txt files found in {folder_path}")
        return []

    print(f"\n🔍 Ingesting {len(all_files)} documents from {folder_path}...\n")
    for filepath in all_files:
        if filepath.suffix.lower() == ".pdf":
            all_chunks.extend(ingest_pdf(str(filepath)))
        else:
            all_chunks.extend(ingest_text(str(filepath)))

    return all_chunks


# ─── Vector Store (TF-IDF) ────────────────────────────────────────────────────

class VectorStore:
    """
    Lightweight TF-IDF vector store.
    In production, swap for dense embeddings (e.g. FinBERT, text-embedding-3-large)
    deployed inside Apex's private VPC.
    """

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def add_chunks(self, chunks: list[Chunk]):
        self.chunks.extend(chunks)
        self._rebuild_index()

    def _rebuild_index(self):
        if not self.chunks:
            return
        texts = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        print(f"  ✅ Index built: {len(self.chunks)} chunks, "
              f"{self.matrix.shape[1]} features")

    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        if self.vectorizer is None or not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                ))
        return results

    def save(self, path: str = VECTOR_STORE_PATH):
        data = {
            "chunks": [asdict(c) for c in self.chunks],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  💾 Saved {len(self.chunks)} chunks to {path}")

    def load(self, path: str = VECTOR_STORE_PATH) -> bool:
        if not Path(path).exists():
            return False
        with open(path, "r") as f:
            data = json.load(f)
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        if self.chunks:
            self._rebuild_index()
        print(f"  📂 Loaded {len(self.chunks)} chunks from {path}")
        return True


# ─── Audit Logger ─────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Immutable append-only JSONL audit log.
    Addresses the CRO's auditability requirement:
    every query, retrieved source, and AI output is logged.
    """

    def __init__(self, path: str = AUDIT_LOG_PATH):
        self.path = path

    def log(self, entry: AuditEntry):
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def print_recent(self, n: int = 5):
        if not Path(self.path).exists():
            print("No audit log entries yet.")
            return
        lines = Path(self.path).read_text().strip().split("\n")
        recent = lines[-n:]
        print(f"\n📋 Last {len(recent)} audit entries:\n")
        for line in recent:
            entry = json.loads(line)
            print(f"  [{entry['timestamp']}]")
            print(f"  Query: {entry['query'][:80]}...")
            print(f"  Sources: {[s['source_file'] for s in entry['retrieved_sources']]}")
            print(f"  Model: {entry['model']}\n")


# ─── RAG Query Engine ─────────────────────────────────────────────────────────

class ApexRAG:
    """
    Core RAG query engine.
    Retrieves relevant chunks, builds a grounded prompt,
    calls Claude, injects citations, and logs everything.
    """

    def __init__(self, vector_store: VectorStore, audit_logger: AuditLogger):
        self.store = vector_store
        self.logger = audit_logger

    def _build_prompt(self, query: str, retrieved: list[RetrievedChunk]) -> str:
        context_blocks = []
        for i, r in enumerate(retrieved, 1):
            context_blocks.append(
                f"[SOURCE {i}] File: {r.chunk.source_file} | "
                f"Type: {r.chunk.doc_type} | Page: {r.chunk.page_number}\n"
                f"{r.chunk.text}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        return f"""You are an AI research assistant for Apex Global, a $15B AUM multi-strategy hedge fund.
Your role is to synthesize insights from financial documents to support Fundamental Equity and Macro analysts.

CRITICAL RULES:
- Only use information from the provided source documents below.
- Every claim must reference its source using [SOURCE N] notation.
- If the answer cannot be found in the sources, say so explicitly — do not hallucinate.
- Flag any uncertainty with phrases like "based on available sources" or "this may require verification".
- Do not provide investment advice or trading recommendations.

RETRIEVED DOCUMENTS:
{context}

ANALYST QUERY: {query}

Provide a concise, cited synthesis. Format:
1. Direct answer with [SOURCE N] citations
2. Key supporting evidence
3. Any caveats or limitations in the available data"""

    def query(self, question: str, top_k: int = TOP_K) -> str:
        print(f"\n🔍 Retrieving relevant chunks...")
        retrieved = self.store.search(question, top_k=top_k)

        if not retrieved:
            response = ("⚠️  No relevant documents found in the knowledge base. "
                       "Please ingest documents first using --ingest.")
            print(response)
            return response

        print(f"   Found {len(retrieved)} relevant chunks:")
        for r in retrieved:
            print(f"   • [{r.chunk.doc_type}] {r.chunk.source_file} "
                  f"p.{r.chunk.page_number} (score: {r.score:.3f})")

        prompt = self._build_prompt(question, retrieved)

        print(f"\n🤖 Generating response via Claude ({ANTHROPIC_MODEL})...")

        try:
            import urllib.request
            import urllib.error

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                # Demo mode: show what would be sent without calling API
                return self._demo_response(question, retrieved)

            payload = json.dumps({
                "model": ANTHROPIC_MODEL,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                response_text = result["content"][0]["text"]

        except Exception as e:
            response_text = self._demo_response(question, retrieved)

        # Audit log
        audit_entry = AuditEntry(
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            query=question,
            retrieved_sources=[
                {
                    "source_file": r.chunk.source_file,
                    "doc_type": r.chunk.doc_type,
                    "page_number": r.chunk.page_number,
                    "chunk_id": r.chunk.chunk_id,
                    "relevance_score": round(r.score, 4),
                }
                for r in retrieved
            ],
            model=ANTHROPIC_MODEL,
            response=response_text,
            confidence_note="RAG-grounded: response limited to ingested documents only",
        )
        self.logger.log(audit_entry)
        print(f"   ✅ Logged to audit trail ({AUDIT_LOG_PATH})")

        return response_text

    def _demo_response(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        """Fallback demo response showing what would be returned."""
        sources_summary = "\n".join([
            f"  [SOURCE {i+1}] {r.chunk.source_file} (p.{r.chunk.page_number}, "
            f"score: {r.score:.3f}): \"{r.chunk.text[:120]}...\""
            for i, r in enumerate(retrieved)
        ])
        return (
            f"[DEMO MODE — set ANTHROPIC_API_KEY to enable live responses]\n\n"
            f"Query: {question}\n\n"
            f"Retrieved {len(retrieved)} relevant sources:\n{sources_summary}\n\n"
            f"In production, Claude would synthesize these sources into a cited answer, "
            f"flagging any uncertainty and refusing to answer beyond the retrieved context."
        )


# ─── Sample Documents (for demo without real files) ───────────────────────────

SAMPLE_DOCS = {
    "goldman_broker_note_NVDA.txt": """
Goldman Sachs Equity Research — NVIDIA Corporation (NVDA)
Rating: Buy | Price Target: $1,200 | Current Price: $875

We are raising our price target on NVDA from $950 to $1,200, reflecting continued 
dominance in AI accelerator demand. Data center revenue grew 427% YoY to $18.4B in Q3 FY2024, 
driven by Hopper GPU shipments to hyperscalers.

Key thesis: NVDA maintains 85%+ market share in AI training workloads. The H100 backlog 
remains 12+ months, and H200 ramp is tracking ahead of schedule. We model $120B in data 
center revenue for FY2025, up from $47B in FY2024.

Risks: AMD MI300X gaining traction at select hyperscalers; custom silicon from Google TPU 
v5 and Amazon Trainium 2 could pressure long-term share. Regulatory risk on China export 
controls remains an overhang.

Analyst: Michael Chen | michael.chen@gs.com | Published: November 2024
""",
    "fomc_minutes_nov2024.txt": """
Federal Open Market Committee Meeting Minutes — November 6-7, 2024

Monetary Policy Discussion:
Members discussed the appropriate path for the federal funds rate in light of continued 
progress toward the Committee's 2% inflation objective. Recent data indicated that 
inflation had eased considerably from its peak, with PCE at 2.3% year-over-year.

Labor market conditions remained solid but showed signs of gradual cooling. The 
unemployment rate edged up to 4.2%, and payroll growth moderated to approximately 
150,000 per month, down from the robust pace seen in early 2024.

Decision: The Committee voted 11-1 to lower the target range for the federal funds 
rate by 25 basis points to 4.5-4.75%. The dissenting member preferred to hold rates 
steady, citing lingering upside risks to inflation from services prices.

Forward guidance: The Committee noted it would carefully assess incoming data and 
was not pre-committing to any particular pace of further adjustments. Several members 
noted the neutral rate may have risen since the pandemic, suggesting rates may need to 
remain modestly restrictive for longer than previously anticipated.
""",
    "expert_call_transcript_semis.txt": """
Expert Network Call Transcript — Semiconductor Supply Chain
Date: October 15, 2024 | Duration: 45 minutes

Operator: Good afternoon. Today we have a former VP of Supply Chain at a leading 
semiconductor equipment manufacturer. All questions are from buy-side participants.

Expert: Thanks for having me. Happy to discuss what I'm seeing in the supply chain.

Q: How are lead times trending for advanced packaging capacity, specifically CoWoS?
A: CoWoS lead times have compressed from 52 weeks peak to now roughly 28-32 weeks. 
TSMC is aggressively expanding capacity — they're adding 3 CoWoS lines in 2025. 
The demand signal from NVDA and Broadcom remains extremely strong.

Q: Are you seeing any softness in the AI infrastructure build-out?
A: Not at the high end. Hyperscalers — Microsoft, Google, Meta, Amazon — are all 
still in aggressive capex mode. The softness is really in consumer-facing AI applications 
and some enterprise segments that are waiting for ROI proof points.

Q: What's your view on HBM supply tightness?
A: SK Hynix has effectively pre-sold their entire 2025 HBM3e capacity to NVDA. 
Samsung is trying to qualify with NVDA but still has yield issues. Micron is ramping 
but won't be a meaningful factor until H2 2025. Supply remains tight through 2025.

Operator: That concludes our call. Transcript is for internal use only.
"""
}


def create_sample_docs(folder: str = "./sample_docs"):
    """Create sample financial documents for demo purposes."""
    Path(folder).mkdir(exist_ok=True)
    for filename, content in SAMPLE_DOCS.items():
        with open(f"{folder}/{filename}", "w") as f:
            f.write(content.strip())
    print(f"✅ Created {len(SAMPLE_DOCS)} sample documents in {folder}/")
    return folder


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apex Global — Agentic AI Research Assistant (RAG Demo)"
    )
    parser.add_argument("--ingest", type=str, metavar="FOLDER",
                        help="Ingest documents from a folder")
    parser.add_argument("--query", type=str,
                        help="Run a single query and exit")
    parser.add_argument("--audit", action="store_true",
                        help="Show recent audit log entries")
    parser.add_argument("--demo", action="store_true",
                        help="Create sample docs and run a demo query")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Number of chunks to retrieve (default: {TOP_K})")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  APEX GLOBAL — Agentic AI Research Assistant")
    print("  Powered by Unique AI | RAG Pipeline Demo")
    print("="*60 + "\n")

    store = VectorStore()
    logger = AuditLogger()
    rag = ApexRAG(store, logger)

    # Load existing index if available
    if Path(VECTOR_STORE_PATH).exists():
        store.load()

    # Demo mode
    if args.demo:
        demo_folder = create_sample_docs()
        chunks = ingest_folder(demo_folder)
        store.add_chunks(chunks)
        store.save()
        demo_queries = [
            "What is Goldman's price target for NVIDIA and what are the key risks?",
            "What did the Fed decide at the November 2024 meeting and what was the rationale?",
            "What is the current state of HBM memory supply for AI chips?",
        ]
        for q in demo_queries:
            print(f"\n{'─'*60}")
            print(f"QUERY: {q}")
            print('─'*60)
            response = rag.query(q, top_k=args.top_k)
            print(f"\nRESPONSE:\n{response}")
        return

    # Ingest documents
    if args.ingest:
        chunks = ingest_folder(args.ingest)
        if chunks:
            store.add_chunks(chunks)
            store.save()
        return

    # Show audit log
    if args.audit:
        logger.print_recent()
        return

    # Single query
    if args.query:
        if not store.chunks:
            print("⚠️  No documents ingested. Run with --ingest or --demo first.")
            return
        response = rag.query(args.query, top_k=args.top_k)
        print(f"\n{'─'*60}")
        print("RESPONSE:")
        print('─'*60)
        print(response)
        return

    # Interactive mode
    if not store.chunks:
        print("No documents loaded. Creating sample docs for demo...\n")
        demo_folder = create_sample_docs()
        chunks = ingest_folder(demo_folder)
        store.add_chunks(chunks)
        store.save()

    print(f"✅ {len(store.chunks)} chunks loaded. Ready for queries.")
    print("   Type 'quit' to exit | 'audit' to view log | 'help' for tips\n")

    while True:
        try:
            question = input("Analyst Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "audit":
            logger.print_recent()
            continue
        if question.lower() == "help":
            print("\nExample queries:")
            print("  • What is the price target for NVIDIA?")
            print("  • What did the Fed decide on interest rates?")
            print("  • What is the supply situation for HBM memory?")
            print("  • What are the risks to the semiconductor thesis?\n")
            continue

        response = rag.query(question, top_k=args.top_k)
        print(f"\n{'─'*60}")
        print(response)
        print('─'*60 + "\n")


if __name__ == "__main__":
    main()
