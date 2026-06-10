"""
Apex Global — Document Delta Detector
======================================
Compares two versions of a financial document and surfaces:
  • Added / removed / changed sentences
  • Tone shift analysis (hawkish/dovish, bullish/bearish, risk language)
  • Section-level change summary
  • Alpha signal flags (material language changes worth analyst attention)
  • HTML report with highlighted diff

Supports: CB Minutes, 10-K/Regulatory Filings, Broker Research Notes

Usage:
    python delta_detector.py doc_v1.txt doc_v2.txt
    python delta_detector.py doc_v1.pdf doc_v2.pdf --type cb_minutes
    python delta_detector.py doc_v1.txt doc_v2.txt --serve       # opens browser UI
    python delta_detector.py --demo                               # runs with sample docs

Requirements:
    pip install anthropic pypdf scikit-learn numpy
"""

import os
import re
import json
import difflib
import hashlib
import argparse
import datetime
import webbrowser
import http.server
import threading
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import pypdf


# ─── Tone Signal Dictionaries ─────────────────────────────────────────────────

TONE_SIGNALS = {
    "cb_minutes": {
        "hawkish": [
            "upside risks", "inflation remains elevated", "further tightening",
            "restrictive", "vigilant", "persistent inflation", "above target",
            "labour market tight", "wage growth", "rate increases", "higher for longer",
        ],
        "dovish": [
            "downside risks", "inflation easing", "eased considerably", "cooling",
            "moderated", "below target", "rate cuts", "accommodative", "slowing growth",
            "disinflation", "labour market softening", "gradual adjustment",
        ],
    },
    "filing": {
        "risk_increase": [
            "new risk", "material uncertainty", "significant risk", "adversely affect",
            "could result in", "cannot assure", "no assurance", "litigation",
            "regulatory scrutiny", "impairment", "going concern",
        ],
        "risk_decrease": [
            "risk mitigated", "resolved", "no longer", "reduced exposure",
            "settled", "withdrawn", "compliance achieved",
        ],
    },
    "broker_note": {
        "bullish": [
            "upgrade", "buy", "outperform", "overweight", "price target increase",
            "strong demand", "beat expectations", "raised guidance", "accelerating",
            "market share gains", "margin expansion",
        ],
        "bearish": [
            "downgrade", "sell", "underperform", "underweight", "price target cut",
            "demand weakness", "missed expectations", "lowered guidance", "decelerating",
            "margin pressure", "competitive headwinds",
        ],
    },
}

# Universal high-importance signal words across all doc types
ALPHA_SIGNALS = [
    "for the first time", "unprecedented", "material change", "significant shift",
    "no longer", "reversed", "contrary to", "unexpectedly", "surprise",
    "guidance withdrawn", "outlook revised", "new position", "exit",
    "dissent", "unanimous", "override", "emergency",
]


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SentenceDiff:
    type: str           # added | removed | changed | unchanged
    old_text: str
    new_text: str
    section: str
    is_alpha_signal: bool
    tone_tags: list[str]


@dataclass
class ToneScore:
    doc_type: str
    v1_scores: dict
    v2_scores: dict
    shifts: list[str]


@dataclass
class DeltaReport:
    doc_type: str
    v1_name: str
    v2_name: str
    timestamp: str
    summary: str
    sentence_diffs: list[SentenceDiff]
    tone_score: ToneScore
    alpha_flags: list[str]
    stats: dict


# ─── Document Loading ─────────────────────────────────────────────────────────

def load_document(filepath: str) -> str:
    path = Path(filepath)
    if path.suffix.lower() == ".pdf":
        text = ""
        with open(filepath, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


def detect_doc_type(text: str, hint: Optional[str] = None) -> str:
    if hint and hint in ("cb_minutes", "filing", "broker_note"):
        return hint
    text_lower = text.lower()
    scores = {
        "cb_minutes": sum(1 for w in [
            "federal reserve", "fomc", "central bank", "monetary policy",
            "basis points", "inflation target", "federal funds rate"
        ] if w in text_lower),
        "filing": sum(1 for w in [
            "10-k", "10-q", "risk factors", "sec filing", "annual report",
            "management discussion", "form 10", "registrant"
        ] if w in text_lower),
        "broker_note": sum(1 for w in [
            "price target", "buy rating", "analyst", "overweight",
            "eps estimate", "equity research", "recommendation"
        ] if w in text_lower),
    }
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "cb_minutes"


# ─── Section Detection ────────────────────────────────────────────────────────

SECTION_PATTERNS = {
    "cb_minutes": [
        r"(staff review|economic outlook|financial conditions|"
        r"participants.{0,20}views|monetary policy|committee policy|"
        r"vote|notation vote|open market)",
    ],
    "filing": [
        r"(item \d|risk factors|management.{0,10}discussion|"
        r"quantitative|financial statements|forward.looking)",
    ],
    "broker_note": [
        r"(investment thesis|valuation|risks|financials|"
        r"recommendation|price target|earnings|outlook)",
    ],
}

def detect_section(text: str, doc_type: str) -> str:
    text_lower = text.lower()[:120]
    patterns = SECTION_PATTERNS.get(doc_type, [])
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(0).title()
    return "General"


# ─── Sentence Splitter ────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    # Split on sentence boundaries but preserve abbreviations
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ─── Tone Analysis ────────────────────────────────────────────────────────────

def score_tone(text: str, doc_type: str) -> dict:
    text_lower = text.lower()
    signals = TONE_SIGNALS.get(doc_type, TONE_SIGNALS["cb_minutes"])
    scores = {}
    for tone, keywords in signals.items():
        scores[tone] = sum(1 for kw in keywords if kw in text_lower)
    return scores


def detect_tone_shifts(v1_scores: dict, v2_scores: dict) -> list[str]:
    shifts = []
    for tone, v2_count in v2_scores.items():
        v1_count = v1_scores.get(tone, 0)
        delta = v2_count - v1_count
        if delta >= 2:
            shifts.append(f"⬆ More {tone} language (+{delta} signals)")
        elif delta <= -2:
            shifts.append(f"⬇ Less {tone} language ({delta} signals)")
    return shifts if shifts else ["No significant tone shift detected"]


# ─── Alpha Signal Detection ───────────────────────────────────────────────────

def check_alpha_signals(sentence: str) -> bool:
    s = sentence.lower()
    return any(signal in s for signal in ALPHA_SIGNALS)


def tag_tone(sentence: str, doc_type: str) -> list[str]:
    s = sentence.lower()
    tags = []
    signals = TONE_SIGNALS.get(doc_type, {})
    for tone, keywords in signals.items():
        if any(kw in s for kw in keywords):
            tags.append(tone)
    return tags


# ─── Core Diff Engine ─────────────────────────────────────────────────────────

def compute_diff(v1_text: str, v2_text: str, doc_type: str) -> list[SentenceDiff]:
    v1_sentences = split_sentences(v1_text)
    v2_sentences = split_sentences(v2_text)

    matcher = difflib.SequenceMatcher(
        None, v1_sentences, v2_sentences, autojunk=False
    )

    diffs = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for s in v1_sentences[i1:i2]:
                diffs.append(SentenceDiff(
                    type="unchanged",
                    old_text=s, new_text=s,
                    section=detect_section(s, doc_type),
                    is_alpha_signal=False,
                    tone_tags=[],
                ))
        elif op == "insert":
            for s in v2_sentences[j1:j2]:
                diffs.append(SentenceDiff(
                    type="added",
                    old_text="", new_text=s,
                    section=detect_section(s, doc_type),
                    is_alpha_signal=check_alpha_signals(s),
                    tone_tags=tag_tone(s, doc_type),
                ))
        elif op == "delete":
            for s in v1_sentences[i1:i2]:
                diffs.append(SentenceDiff(
                    type="removed",
                    old_text=s, new_text="",
                    section=detect_section(s, doc_type),
                    is_alpha_signal=check_alpha_signals(s),
                    tone_tags=tag_tone(s, doc_type),
                ))
        elif op == "replace":
            # Pair up old/new as changes
            old_chunk = v1_sentences[i1:i2]
            new_chunk = v2_sentences[j1:j2]
            max_len = max(len(old_chunk), len(new_chunk))
            for k in range(max_len):
                old_s = old_chunk[k] if k < len(old_chunk) else ""
                new_s = new_chunk[k] if k < len(new_chunk) else ""
                text_for_signal = new_s or old_s
                diffs.append(SentenceDiff(
                    type="changed",
                    old_text=old_s, new_text=new_s,
                    section=detect_section(text_for_signal, doc_type),
                    is_alpha_signal=check_alpha_signals(text_for_signal),
                    tone_tags=tag_tone(text_for_signal, doc_type),
                ))
    return diffs


# ─── AI Summary (Claude) ──────────────────────────────────────────────────────

def generate_ai_summary(diffs: list[SentenceDiff], doc_type: str,
                         v1_name: str, v2_name: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _demo_summary(diffs, doc_type, v1_name, v2_name)

    # Build change digest for Claude
    changes = []
    for d in diffs:
        if d.type == "added":
            changes.append(f"ADDED: {d.new_text}")
        elif d.type == "removed":
            changes.append(f"REMOVED: {d.old_text}")
        elif d.type == "changed":
            changes.append(f"CHANGED FROM: {d.old_text}\n       TO: {d.new_text}")

    if not changes:
        return "No material changes detected between the two documents."

    changes_text = "\n\n".join(changes[:40])  # cap at 40 changes

    doc_type_label = {
        "cb_minutes": "central bank meeting minutes",
        "filing": "regulatory filing (10-K/10-Q)",
        "broker_note": "broker research note",
    }.get(doc_type, "financial document")

    prompt = f"""You are a senior analyst at a $15B AUM hedge fund reviewing changes between two versions of a {doc_type_label}.

Document V1: {v1_name}
Document V2: {v2_name}

The following changes were detected:

{changes_text}

Provide a concise analyst-grade summary covering:
1. HEADLINE: One sentence capturing the most important change
2. KEY SHIFTS: 2-3 bullet points on material changes and their investment implications
3. TONE CHANGE: Did the language become more hawkish/dovish, bullish/bearish, or more/less cautious?
4. ALPHA FLAGS: Any language changes that could affect investment decisions
5. WATCH: What should the analyst monitor as a follow-up?

Be specific, cite the actual language changes, and focus on investment relevance."""

    try:
        import urllib.request
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 800,
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
            return result["content"][0]["text"]
    except Exception as e:
        return _demo_summary(diffs, doc_type, v1_name, v2_name)


def _demo_summary(diffs: list[SentenceDiff], doc_type: str,
                  v1_name: str, v2_name: str) -> str:
    added = [d for d in diffs if d.type == "added"]
    removed = [d for d in diffs if d.type == "removed"]
    changed = [d for d in diffs if d.type == "changed"]
    alpha = [d for d in diffs if d.is_alpha_signal]

    return f"""[DEMO MODE — set ANTHROPIC_API_KEY for AI-generated analysis]

HEADLINE: {len(changed)} sentence(s) changed, {len(added)} added, {len(removed)} removed between {v1_name} and {v2_name}.

KEY SHIFTS:
• {len(changed)} modified passages detected — review highlighted changes below
• {len(added)} new sentences added to V2
• {len(removed)} sentences removed from V1

ALPHA FLAGS: {len(alpha)} sentence(s) contain high-importance signal language
{chr(10).join(f'  • {d.new_text or d.old_text}' for d in alpha[:3])}

WATCH: Review all highlighted sections marked in red/green for investment-relevant language changes.
Set ANTHROPIC_API_KEY to enable full AI-powered analyst commentary."""


# ─── HTML Report Generator ────────────────────────────────────────────────────

def generate_html_report(report: DeltaReport) -> str:
    def diff_row(d: SentenceDiff) -> str:
        if d.type == "unchanged":
            return ""

        tone_badges = "".join(
            f'<span class="badge badge-{t}">{t}</span>'
            for t in d.tone_tags
        )
        alpha_badge = '<span class="badge badge-alpha">⚡ ALPHA SIGNAL</span>' \
                      if d.is_alpha_signal else ""

        if d.type == "added":
            return f"""
            <tr class="row-added">
                <td class="cell-type">➕ Added</td>
                <td class="cell-old">—</td>
                <td class="cell-new">{d.new_text}</td>
                <td class="cell-meta">{d.section}<br>{tone_badges}{alpha_badge}</td>
            </tr>"""
        elif d.type == "removed":
            return f"""
            <tr class="row-removed">
                <td class="cell-type">➖ Removed</td>
                <td class="cell-old">{d.old_text}</td>
                <td class="cell-new">—</td>
                <td class="cell-meta">{d.section}<br>{tone_badges}{alpha_badge}</td>
            </tr>"""
        else:
            return f"""
            <tr class="row-changed">
                <td class="cell-type">✏️ Changed</td>
                <td class="cell-old">{d.old_text}</td>
                <td class="cell-new">{d.new_text}</td>
                <td class="cell-meta">{d.section}<br>{tone_badges}{alpha_badge}</td>
            </tr>"""

    # Build tone shift pills
    tone_shift_html = "".join(
        f'<div class="tone-pill">{s}</div>'
        for s in report.tone_score.shifts
    )

    # Build alpha flags
    alpha_html = "".join(
        f'<li>{flag}</li>' for flag in report.alpha_flags
    ) if report.alpha_flags else "<li>No alpha signals detected</li>"

    # Build diff rows (skip unchanged for cleaner view)
    rows_html = "".join(
        diff_row(d) for d in report.sentence_diffs
        if d.type != "unchanged"
    )

    if not rows_html:
        rows_html = """<tr><td colspan="4" style="text-align:center;padding:2rem;color:#64748b;">
            No material differences detected between documents.</td></tr>"""

    summary_html = report.summary.replace("\n", "<br>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Document Delta Report — {report.v1_name} vs {report.v2_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f0f4f8; color: #1e2761; }}

  .header {{ background: #1e2761; color: white; padding: 1.5rem 2rem; }}
  .header h1 {{ font-size: 1.4rem; font-weight: 700; }}
  .header .sub {{ color: #cadcfc; font-size: 0.85rem; margin-top: 0.3rem; }}
  .header .badge-type {{ background: #028090; color: white; padding: 0.2rem 0.7rem;
    border-radius: 20px; font-size: 0.75rem; margin-left: 0.5rem; }}

  .container {{ max-width: 1400px; margin: 0 auto; padding: 1.5rem; }}

  .stats-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
                margin-bottom: 1.5rem; }}
  .stat-card {{ background: white; border-radius: 10px; padding: 1rem 1.2rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .stat-card .num {{ font-size: 2rem; font-weight: 700; }}
  .stat-card .label {{ font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }}
  .stat-added .num {{ color: #059669; }}
  .stat-removed .num {{ color: #dc2626; }}
  .stat-changed .num {{ color: #f59e0b; }}
  .stat-alpha .num {{ color: #7c3aed; }}

  .panel {{ background: white; border-radius: 10px; padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1.5rem; }}
  .panel h2 {{ font-size: 1rem; font-weight: 700; color: #1e2761;
               border-bottom: 2px solid #028090; padding-bottom: 0.5rem;
               margin-bottom: 1rem; }}

  .summary-text {{ line-height: 1.7; font-size: 0.9rem; color: #334155; }}

  .tone-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
  .tone-pill {{ background: #eff6ff; border: 1px solid #cadcfc; color: #1e2761;
                padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.82rem; }}

  .alpha-list {{ list-style: none; }}
  .alpha-list li {{ padding: 0.4rem 0; border-bottom: 1px solid #f1f5f9;
                    font-size: 0.88rem; color: #334155; }}
  .alpha-list li:before {{ content: "⚡ "; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 0.83rem; }}
  th {{ background: #1e2761; color: white; padding: 0.7rem 0.8rem;
        text-align: left; font-weight: 600; }}
  td {{ padding: 0.7rem 0.8rem; vertical-align: top;
        border-bottom: 1px solid #f1f5f9; }}
  .cell-type {{ width: 90px; font-weight: 600; white-space: nowrap; }}
  .cell-old {{ width: 35%; color: #64748b; }}
  .cell-new {{ width: 35%; }}
  .cell-meta {{ width: 15%; font-size: 0.78rem; color: #64748b; }}

  .row-added {{ background: #f0fdf4; }}
  .row-added .cell-new {{ color: #065f46; font-weight: 500; }}
  .row-removed {{ background: #fef2f2; }}
  .row-removed .cell-old {{ color: #991b1b; text-decoration: line-through; }}
  .row-changed {{ background: #fffbeb; }}
  .row-changed .cell-old {{ color: #92400e; text-decoration: line-through; }}
  .row-changed .cell-new {{ color: #065f46; font-weight: 500; }}

  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 10px;
            font-size: 0.72rem; font-weight: 600; margin: 0.1rem; }}
  .badge-hawkish {{ background: #fee2e2; color: #991b1b; }}
  .badge-dovish  {{ background: #dbeafe; color: #1e40af; }}
  .badge-bullish {{ background: #d1fae5; color: #065f46; }}
  .badge-bearish {{ background: #fee2e2; color: #991b1b; }}
  .badge-risk_increase {{ background: #fef3c7; color: #92400e; }}
  .badge-risk_decrease {{ background: #d1fae5; color: #065f46; }}
  .badge-alpha   {{ background: #ede9fe; color: #5b21b6; }}

  .footer {{ text-align: center; color: #94a3b8; font-size: 0.78rem;
             padding: 1rem; }}
</style>
</head>
<body>

<div class="header">
  <h1>Document Delta Report
    <span class="badge-type">{report.doc_type.replace("_", " ").title()}</span>
  </h1>
  <div class="sub">
    {report.v1_name} → {report.v2_name} &nbsp;·&nbsp; Generated {report.timestamp}
  </div>
</div>

<div class="container">

  <div class="stats-row">
    <div class="stat-card stat-added">
      <div class="num">{report.stats.get("added", 0)}</div>
      <div class="label">Sentences Added</div>
    </div>
    <div class="stat-card stat-removed">
      <div class="num">{report.stats.get("removed", 0)}</div>
      <div class="label">Sentences Removed</div>
    </div>
    <div class="stat-card stat-changed">
      <div class="num">{report.stats.get("changed", 0)}</div>
      <div class="label">Sentences Changed</div>
    </div>
    <div class="stat-card stat-alpha">
      <div class="num">{report.stats.get("alpha_signals", 0)}</div>
      <div class="label">Alpha Signals</div>
    </div>
  </div>

  <div class="panel">
    <h2>🤖 AI Analyst Summary</h2>
    <div class="summary-text">{summary_html}</div>
  </div>

  <div class="panel">
    <h2>📊 Tone Shift Analysis</h2>
    <div class="tone-row">{tone_shift_html}</div>
  </div>

  <div class="panel">
    <h2>⚡ Alpha Signal Flags</h2>
    <ul class="alpha-list">{alpha_html}</ul>
  </div>

  <div class="panel">
    <h2>📄 Sentence-Level Changes</h2>
    <table>
      <thead>
        <tr>
          <th>Change</th>
          <th>V1 (Before)</th>
          <th>V2 (After)</th>
          <th>Section / Tags</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

</div>
<div class="footer">
  Unique AI · Apex Global Research Assistant · Audit ID: {report.timestamp}
</div>
</body>
</html>"""


# ─── Sample Documents ─────────────────────────────────────────────────────────

SAMPLE_V1 = """Federal Open Market Committee Meeting Minutes — September 2024

The Committee discussed recent economic developments and the outlook for monetary policy.
Members noted that inflation had declined significantly from its peak but remained above the 2 percent target.
The labor market remained strong, with unemployment at 4.0 percent and steady payroll growth.
Several members expressed concern about persistent services inflation.
The Committee voted unanimously to hold the federal funds rate at 5.25-5.50 percent.
Members agreed that further evidence of disinflation would be needed before adjusting the policy rate.
The staff noted upside risks to the inflation outlook from strong consumer spending.
Members emphasized they would remain vigilant and data-dependent in their approach."""

SAMPLE_V2 = """Federal Open Market Committee Meeting Minutes — November 2024

The Committee discussed recent economic developments and the outlook for monetary policy.
Members noted that inflation had eased considerably and was making progress toward the 2 percent target.
The labor market showed signs of gradual cooling, with unemployment edging up to 4.2 percent and payroll growth moderating.
Services inflation had moderated more than expected, reducing concerns among several members.
The Committee voted 11-1 to lower the federal funds rate by 25 basis points to 4.75-5.00 percent.
One member dissented, preferring to hold rates steady citing lingering upside inflation risks.
The staff noted that downside risks to the growth outlook had increased.
Members agreed the neutral rate may have risen, suggesting rates could remain modestly restrictive for longer.
Several members noted they were no longer pre-committing to any particular pace of further adjustments."""


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_delta(v1_path: str, v2_path: str,
              doc_type_hint: Optional[str] = None,
              serve: bool = False,
              output_path: Optional[str] = None) -> DeltaReport:

    print(f"\n{'='*60}")
    print("  APEX GLOBAL — Document Delta Detector")
    print("  Powered by Unique AI")
    print(f"{'='*60}\n")

    print(f"📄 Loading V1: {v1_path}")
    v1_text = load_document(v1_path)

    print(f"📄 Loading V2: {v2_path}")
    v2_text = load_document(v2_path)

    doc_type = detect_doc_type(v1_text + v2_text, doc_type_hint)
    print(f"🔍 Detected document type: {doc_type.replace('_', ' ').title()}")

    print("⚙️  Computing sentence-level diff...")
    diffs = compute_diff(v1_text, v2_text, doc_type)

    # Stats
    stats = {
        "added": sum(1 for d in diffs if d.type == "added"),
        "removed": sum(1 for d in diffs if d.type == "removed"),
        "changed": sum(1 for d in diffs if d.type == "changed"),
        "unchanged": sum(1 for d in diffs if d.type == "unchanged"),
        "alpha_signals": sum(1 for d in diffs if d.is_alpha_signal),
    }
    print(f"   ✅ {stats['added']} added · {stats['removed']} removed · "
          f"{stats['changed']} changed · {stats['alpha_signals']} alpha signals")

    print("📊 Analysing tone shift...")
    v1_tone = score_tone(v1_text, doc_type)
    v2_tone = score_tone(v2_text, doc_type)
    tone_shifts = detect_tone_shifts(v1_tone, v2_tone)
    tone_score = ToneScore(
        doc_type=doc_type,
        v1_scores=v1_tone,
        v2_scores=v2_tone,
        shifts=tone_shifts,
    )
    for shift in tone_shifts:
        print(f"   {shift}")

    # Alpha flags
    alpha_flags = [
        (d.new_text or d.old_text)
        for d in diffs
        if d.is_alpha_signal and d.type != "unchanged"
    ]

    print("🤖 Generating AI analyst summary...")
    summary = generate_ai_summary(
        diffs, doc_type,
        Path(v1_path).name, Path(v2_path).name
    )

    report = DeltaReport(
        doc_type=doc_type,
        v1_name=Path(v1_path).name,
        v2_name=Path(v2_path).name,
        timestamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        summary=summary,
        sentence_diffs=diffs,
        tone_score=tone_score,
        alpha_flags=alpha_flags,
        stats=stats,
    )

    # Print CLI summary
    print(f"\n{'─'*60}")
    print("ANALYST SUMMARY:")
    print('─'*60)
    print(summary)
    print('─'*60)

    if tone_shifts:
        print("\nTONE SHIFTS:")
        for s in tone_shifts:
            print(f"  {s}")

    if alpha_flags:
        print(f"\n⚡ ALPHA SIGNALS ({len(alpha_flags)}):")
        for flag in alpha_flags[:5]:
            print(f"  • {flag[:120]}")

    # Generate HTML
    html = generate_html_report(report)
    out_path = output_path or "delta_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📊 HTML report saved: {out_path}")

    if serve:
        _serve_report(out_path)

    return report


def _serve_report(html_path: str, port: int = 8765):
    """Serve the HTML report in the browser."""
    directory = str(Path(html_path).parent.absolute())
    filename = Path(html_path).name

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, format, *args):
            pass  # suppress request logs

    url = f"http://localhost:{port}/{filename}"
    print(f"\n🌐 Opening report in browser: {url}")
    print("   Press Ctrl+C to stop the server\n")

    threading.Thread(
        target=lambda: webbrowser.open(url),
        daemon=True
    ).start()

    with http.server.HTTPServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apex Global — Document Delta Detector"
    )
    parser.add_argument("v1", nargs="?", help="Path to V1 document (PDF or TXT)")
    parser.add_argument("v2", nargs="?", help="Path to V2 document (PDF or TXT)")
    parser.add_argument("--type", choices=["cb_minutes", "filing", "broker_note"],
                        help="Document type (auto-detected if not specified)")
    parser.add_argument("--serve", action="store_true",
                        help="Open HTML report in browser after generation")
    parser.add_argument("--output", type=str, default="delta_report.html",
                        help="Output HTML file path")
    parser.add_argument("--demo", action="store_true",
                        help="Run with built-in Fed minutes sample documents")
    args = parser.parse_args()

    if args.demo:
        # Write sample docs to temp files
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False) as f1:
            f1.write(SAMPLE_V1)
            v1_path = f1.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False) as f2:
            f2.write(SAMPLE_V2)
            v2_path = f2.name

        run_delta(
            v1_path, v2_path,
            doc_type_hint="cb_minutes",
            serve=args.serve,
            output_path=args.output,
        )
        os.unlink(v1_path)
        os.unlink(v2_path)
        return

    if not args.v1 or not args.v2:
        parser.print_help()
        print("\nTip: Run with --demo to see it in action immediately")
        return

    run_delta(
        args.v1, args.v2,
        doc_type_hint=args.type,
        serve=args.serve,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
