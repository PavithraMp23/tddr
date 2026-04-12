
from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

# Make sure the project root is on sys.path when run directly.
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from user_input_module import process_query

# ---------------------------------------------------------------------------
# RAG module — imported lazily so the UI still works even if deps are missing
# ---------------------------------------------------------------------------
_rag_ready = False
_rag_store = None        # VectorStore singleton
_rag_chunks = []         # all ingested chunks (for retrieval)
_rag_retriever = None    # TemporalRAGRetriever singleton

# ---------------------------------------------------------------------------
# LLM module (Module 3) — imported lazily
# ---------------------------------------------------------------------------
_llm_ready = False

# ---------------------------------------------------------------------------
# Output module (Module 4) — imported lazily
# ---------------------------------------------------------------------------
_output_ready = False

# ---------------------------------------------------------------------------
# Dummy IPC regulation data — pre-loaded at startup for end-to-end demo
# ---------------------------------------------------------------------------
_DUMMY_DOCS = [
    {
        "regulation": "IPC", "version": "1860",
        "effective_from": "1860-01-01", "effective_to": "2017-12-31",
        "text": (
            "Section 21 Public Servant\n"
            "The words 'public servant' denote a person falling under any of the descriptions hereinafter following, "
            "including every officer in the service or pay of the Government or of any local authority.\n\n"
            "Section 302 Punishment for Murder\n"
            "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.\n\n"
            "Section 375 Rape\n"
            "A man is said to commit rape who, except in the case hereinafter excepted, has sexual intercourse with a woman "
            "under the circumstances of force, coercion, or without consent. The punishment shall be rigorous imprisonment "
            "of not less than seven years, which may extend to imprisonment for life.\n\n"
            "Section 124A Sedition\n"
            "Whoever by words, either spoken or written, or by signs, or by visible representation, or otherwise, "
            "brings or attempts to bring into hatred or contempt, or excites or attempts to excite disaffection towards "
            "the Government established by law in India, shall be punished with imprisonment for life.\n\n"
            "Section 498A Cruelty by Husband\n"
            "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty "
            "shall be punished with imprisonment for a term which may extend to three years and shall also be liable to fine."
        ),
    },
    {
        "regulation": "IPC", "version": "2018",
        "effective_from": "2018-01-01", "effective_to": None,
        "text": (
            "Section 21 Public Servant (Amended 2018)\n"
            "The words 'public servant' denote a person falling under any of the descriptions hereinafter following. "
            "This definition has been expanded to include members of public sector corporations and "
            "government-aided institutions receiving central or state funds.\n\n"
            "Section 302 Punishment for Murder (Unchanged)\n"
            "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.\n\n"
            "Section 375 Rape (Amended 2018)\n"
            "A man is said to commit rape who has sexual intercourse with a woman under circumstances of "
            "force, coercion, or without consent. The minimum punishment has been enhanced to rigorous "
            "imprisonment of not less than ten years, which may extend to life imprisonment or death in "
            "cases involving victims below sixteen years of age.\n\n"
            "Section 124A Sedition (Under Review)\n"
            "The provision continues to exist but its constitutional validity was placed under supreme court "
            "scrutiny. Enforcement has been stayed by order of the Supreme Court of India pending re-examination.\n\n"
            "Section 498A Cruelty by Husband (Amended 2018)\n"
            "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty "
            "shall be punished with imprisonment for a term which may extend to three years. "
            "Bail provisions have been modified to require prior approval from the magistrate."
        ),
    },
    {
        "regulation": "IT_ACT", "version": "2008",
        "effective_from": "2008-10-27", "effective_to": "2015-03-24",
        "text": (
            "Section 66A Punishment for sending offensive messages through communication service\n"
            "Any person who sends, by means of a computer resource or a communication device, "
            "any information that is grossly offensive, or has menacing character, or any information "
            "which he knows to be false and causes annoyance, inconvenience, danger, or insult "
            "shall be punishable with imprisonment for a term which may extend to three years and with fine."
        ),
    },
    {
        "regulation": "IT_ACT", "version": "2015",
        "effective_from": "2015-03-25", "effective_to": None,
        "text": (
            "Section 66A — Struck Down\n"
            "Section 66A of the Information Technology Act, 2000 was declared unconstitutional by the "
            "Supreme Court of India in Shreya Singhal v. Union of India on 24 March 2015, "
            "as it violated the fundamental right to freedom of speech and expression guaranteed under "
            "Article 19(1)(a) of the Constitution of India. The section is no longer in force."
        ),
    },
    {
        "regulation": "SWM_RULES", "version": "2000",
        "effective_from": "2000-09-25", "effective_to": "2016-03-31",
        "text": (
            "Rule 3 Municipal Solid Wastes\n"
            "Every municipal authority shall be responsible for the implementation of these rules and for "
            "infrastructure development for collection, storage, segregation, transportation, processing "
            "and disposal of municipal solid wastes.\n\n"
            "Rule 5 Responsibilities of Municipal Authority\n"
            "The municipal authority shall undertake phased programme to ensure the separate collection "
            "and scientific disposal of solid wastes. Prohibition on littering in public places.\n\n"
            "Rule 8 Land for Disposal\n"
            "The State Government or the Union territory administration shall identify and allocate or "
            "earmark land for the purpose of setting up of waste processing and disposal facilities."
        ),
    },
    {
        "regulation": "SWM_RULES", "version": "2016",
        "effective_from": "2016-04-01", "effective_to": "2021-02-28",
        "text": (
            "Rule 4 Duties of Waste Generator\n"
            "Every waste generator shall segregate waste into: (a) wet waste (biodegradable), "
            "(b) dry waste (recyclable) and (c) domestic hazardous waste. Mixed waste shall not "
            "be handed over to the waste collector.\n\n"
            "Rule 5 Duties of Local Bodies\n"
            "No person shall dispose of construction debris or solid waste on public roads, footpaths, "
            "drains, water bodies, or any public place. Construction waste must be processed separately.\n\n"
            "Rule 9 Collection of User Fee\n"
            "The local body shall levy and collect user fees from bulk waste generators and market "
            "associations for processing and disposal of solid waste generated by them.\n\n"
            "Rule 22 Extended Producer Responsibility (EPR)\n"
            "Manufacturers and brand owners shall take steps to minimize generation of waste, "
            "promote recycling, and set up material recovery facilities."
        ),
    },
    {
        "regulation": "SWM_RULES", "version": "2021",
        "effective_from": "2021-03-01", "effective_to": None,
        "text": (
            "Rule 5 Prohibition and Segregation (Amended 2021)\n"
            "Waste generators must segregate waste into a minimum of three streams: wet, dry, and "
            "domestic hazardous. Non-compliance attracts enhanced penalties. Bulk generators must "
            "set up in-situ composting or biomethanation facilities on premises.\n\n"
            "Rule 10 Extended Producer Responsibility (Strengthened)\n"
            "Producers, importers and brand owners must achieve prescribed EPR targets annually. "
            "Any shortfall must be offset through purchase of EPR certificates from registered "
            "waste processors. Non-compliance shall attract proportional environmental compensation.\n\n"
            "Rule 12 Decentralised Processing\n"
            "Urban Local Bodies shall establish decentralized composting facilities at ward level. "
            "Apartment complexes and institutions generating more than 5 kg of biodegradable waste "
            "shall be mandated to process their waste at source.\n\n"
            "Rule 15 Plastic Waste Integration\n"
            "Plastic waste management is fully integrated with SWM Rules 2021 in line with the "
            "Plastic Waste Management Amendment Rules, 2021, prohibiting single-use plastics."
        ),
    },
]


def _init_rag():
    global _rag_ready, _rag_store, _rag_chunks, _rag_retriever
    try:
        import glob as _glob
        from rag_module import (
            VectorStore, TemporalRAGRetriever,
            ingest_document, chunk_document, embed_chunks,
        )
        _rag_store = VectorStore()
        # Pre-load dummy IPC/law data
        for doc_data in _DUMMY_DOCS:
            doc    = ingest_document(doc_data)
            chunks = chunk_document(doc)
            embed_chunks(chunks)
            _rag_chunks.extend(chunks)

        # Ingest real PDFs from the data/ directory
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        if os.path.isdir(data_dir):
            pdf_paths = sorted(_glob.glob(os.path.join(data_dir, "*.pdf")))
            for pdf_path in pdf_paths:
                try:
                    print(f"  [RAG] Ingesting PDF: {os.path.basename(pdf_path)} ...")
                    doc = ingest_document(pdf_path)
                    chunks = chunk_document(doc)
                    embed_chunks(chunks)
                    _rag_chunks.extend(chunks)
                    print(f"  [RAG]   → {len(chunks)} chunks  "
                          f"(regulation={doc.regulation}, version={doc.version})")
                except Exception as pdf_err:
                    print(f"  [RAG]   ✗ Failed: {pdf_err}")

        if _rag_chunks:
            _rag_store.add_chunks(_rag_chunks)
        _rag_retriever = TemporalRAGRetriever(_rag_store, _rag_chunks)
        _rag_ready = True
        print(f"  [RAG] Ready — {len(_rag_chunks)} total chunks in vector store.")
    except Exception as e:
        print(f"  [RAG] Could not initialise (install deps to enable): {e}")
        _rag_ready = False

_init_rag()

# ---------------------------------------------------------------------------
# HTML page (embedded)
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>TDDR — Temporal Regulation Drift Detector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0d1117;
      --surface:  #161b22;
      --border:   #30363d;
      --accent:   #58a6ff;
      --accent2:  #3fb950;
      --warn:     #d29922;
      --muted:    #8b949e;
      --text:     #e6edf3;
      --string:   #a5d6ff;
      --number:   #79c0ff;
      --key:      #ff7b72;
      --bool:     #56d364;
      --null:     #8b949e;
      --purple:   #bc8cff;
      --radius:   10px;
    }

    html, body {
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', sans-serif;
      font-size: 15px;
    }

    /* ── Header ── */
    header {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 18px 32px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .logo {
      font-size: 22px;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: -0.5px;
    }
    .logo-sub { font-size: 12px; color: var(--muted); font-weight: 400; margin-top: 2px; }
    .header-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
    .badge {
      font-size: 11px; color: var(--accent);
      background: rgba(88,166,255,.1);
      border: 1px solid rgba(88,166,255,.25);
      padding: 3px 10px; border-radius: 20px;
      font-family: 'JetBrains Mono', monospace;
    }
    .badge.green {
      color: var(--accent2);
      background: rgba(63,185,80,.1);
      border-color: rgba(63,185,80,.25);
    }

    /* ── Tabs ── */
    .tabs {
      display: flex;
      gap: 0;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      padding: 0 32px;
    }
    .tab-btn {
      padding: 12px 20px;
      font-size: 13px;
      font-weight: 500;
      font-family: 'Inter', sans-serif;
      background: transparent;
      color: var(--muted);
      border: none;
      border-bottom: 2px solid transparent;
      cursor: pointer;
      transition: all .15s;
      display: flex;
      align-items: center;
      gap: 7px;
    }
    .tab-btn:hover { color: var(--text); }
    .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
    .tab-btn .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--muted);
    }
    .tab-btn.active .dot { background: var(--accent); }

    /* ── Main layout ── */
    main {
      max-width: 960px;
      margin: 0 auto;
      padding: 28px 24px;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: contents; }

    /* ── Cards ── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 22px;
    }
    .card-label {
      font-size: 11px; font-weight: 600;
      letter-spacing: .08em; text-transform: uppercase;
      color: var(--muted); margin-bottom: 12px;
    }

    /* ── Inputs ── */
    .query-area, .doc-area {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-family: 'Inter', sans-serif;
      font-size: 14px;
      line-height: 1.6;
      padding: 12px 14px;
      resize: vertical;
      outline: none;
      transition: border-color .2s;
    }
    .query-area { min-height: 72px; font-size: 15px; }
    .doc-area   { min-height: 180px; font-family: 'JetBrains Mono', monospace; font-size: 13px; }
    .query-area:focus, .doc-area:focus { border-color: var(--accent); }
    .query-area::placeholder, .doc-area::placeholder { color: var(--muted); }

    /* ── Example chips ── */
    .examples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .example-chip {
      font-size: 12px; padding: 5px 12px;
      border: 1px solid var(--border); border-radius: 20px;
      cursor: pointer; color: var(--muted);
      background: transparent; transition: all .15s;
      font-family: 'Inter', sans-serif;
    }
    .example-chip:hover { border-color: var(--accent); color: var(--accent); background: rgba(88,166,255,.07); }

    /* ── Two-column layout for RAG query ── */
    .two-col { display: grid; grid-template-columns: 1fr 240px; gap: 10px; align-items: end; }
    .two-col input[type="text"] {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      padding: 10px 14px;
      outline: none;
      transition: border-color .2s;
    }
    .two-col input[type="text"]:focus { border-color: var(--accent); }
    .two-col input::placeholder { color: var(--muted); }
    .field-label { font-size: 11px; color: var(--muted); margin-bottom: 5px; letter-spacing: .05em; text-transform: uppercase; }

    /* ── Buttons ── */
    .btn-row { display: flex; gap: 10px; margin-top: 14px; align-items: center; }
    .btn-submit {
      padding: 9px 22px;
      background: var(--accent); color: #0d1117;
      border: none; border-radius: 8px;
      font-size: 14px; font-weight: 600;
      cursor: pointer; font-family: 'Inter', sans-serif;
      transition: opacity .15s, transform .1s;
    }
    .btn-submit:hover { opacity: .88; }
    .btn-submit:active { transform: scale(.97); }
    .btn-submit:disabled { opacity: .45; cursor: not-allowed; }
    .btn-submit.purple { background: var(--purple); }
    .btn-submit.green  { background: var(--accent2); }

    .btn-clear {
      padding: 9px 16px; background: transparent;
      color: var(--muted); border: 1px solid var(--border);
      border-radius: 8px; font-size: 14px; cursor: pointer;
      font-family: 'Inter', sans-serif; transition: all .15s;
    }
    .btn-clear:hover { color: var(--text); border-color: var(--text); }

    .hint { font-size: 12px; color: var(--muted); margin-left: auto; }

    /* ── Spinner ── */
    .spinner {
      display: none; width: 16px; height: 16px;
      border: 2px solid rgba(88,166,255,.3);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin .7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Pipeline status tags ── */
    .result-header {
      display: flex; align-items: center;
      justify-content: space-between; margin-bottom: 16px;
    }
    .pipeline-tags { display: flex; gap: 6px; flex-wrap: wrap; }
    .tag {
      font-size: 11px; font-family: 'JetBrains Mono', monospace;
      padding: 3px 9px; border-radius: 6px;
      background: rgba(255,255,255,.06); border: 1px solid var(--border);
      color: var(--muted);
    }
    .tag.active { background: rgba(88,166,255,.12); border-color: rgba(88,166,255,.4); color: var(--accent); }
    .tag.ok     { background: rgba(63,185,80,.10);  border-color: rgba(63,185,80,.35); color: var(--accent2); }
    .tag.purple { background: rgba(188,140,255,.10); border-color: rgba(188,140,255,.35); color: var(--purple); }
    .tag.warn   { background: rgba(210,153,34,.10); border-color: rgba(210,153,34,.35); color: var(--warn); }
    .tag.error  { background: rgba(248,81,73,.10);  border-color: rgba(248,81,73,.35);  color: #f85149; }

    .copy-btn {
      font-size: 12px; padding: 5px 12px;
      border: 1px solid var(--border); border-radius: 6px;
      background: transparent; color: var(--muted);
      cursor: pointer; font-family: 'Inter', sans-serif;
      transition: all .15s;
    }
    .copy-btn:hover { color: var(--text); border-color: var(--text); }

    /* ── JSON block ── */
    .json-block {
      background: var(--bg); border: 1px solid var(--border);
      border-radius: 8px; padding: 18px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px; line-height: 1.75;
      overflow-x: auto; white-space: pre;
      min-height: 100px;
    }
    .json-key  { color: var(--key); }
    .json-str  { color: var(--string); }
    .json-num  { color: var(--number); }
    .json-bool { color: var(--bool); }
    .json-null { color: var(--null); }
    .json-punct{ color: var(--muted); }

    /* ── Summary grid ── */
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
      gap: 10px; margin-top: 14px;
    }
    .summary-item {
      background: var(--bg); border: 1px solid var(--border);
      border-radius: 8px; padding: 12px 14px;
    }
    .summary-item .s-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 5px; }
    .summary-item .s-value { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--text); word-break: break-all; }
    .s-value.highlight { color: var(--accent); }
    .s-value.green     { color: var(--accent2); }
    .s-value.purple    { color: var(--purple); }
    .s-value.grey      { color: var(--muted); font-style: italic; }

    /* ── RAG chunk cards ── */
    .chunk-list { display: flex; flex-direction: column; gap: 12px; margin-top: 4px; }
    .chunk-card {
      background: var(--bg); border: 1px solid var(--border);
      border-radius: 8px; padding: 16px;
      position: relative;
    }
    .chunk-card:hover { border-color: rgba(188,140,255,.4); }
    .chunk-meta {
      display: flex; flex-wrap: wrap; gap: 8px;
      margin-bottom: 10px; align-items: center;
    }
    .chunk-rank {
      width: 24px; height: 24px; border-radius: 50%;
      background: rgba(188,140,255,.15);
      border: 1px solid rgba(188,140,255,.35);
      color: var(--purple);
      font-size: 11px; font-weight: 700;
      display: flex; align-items: center; justify-content: center;
    }
    .chunk-text {
      font-size: 13px; line-height: 1.65;
      color: var(--text); white-space: pre-wrap; word-break: break-word;
    }

    /* ── Store status bar ── */
    .store-bar {
      background: var(--bg); border: 1px solid var(--border);
      border-radius: 8px; padding: 12px 16px;
      display: flex; align-items: center; gap: 12px;
      font-size: 13px;
    }
    .store-dot {
      width: 9px; height: 9px; border-radius: 50%;
      background: var(--muted); flex-shrink: 0;
      transition: background .3s;
    }
    .store-dot.loaded { background: var(--accent2); box-shadow: 0 0 8px rgba(63,185,80,.6); }

    /* ── Errors ── */
    .error-msg {
      color: #f85149;
      font-family: 'JetBrains Mono', monospace; font-size: 13px;
      background: rgba(248,81,73,.07);
      border: 1px solid rgba(248,81,73,.25);
      border-radius: 8px; padding: 14px;
    }

    /* ── LLM config select & input ── */
    .llm-select, .llm-input {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-family: 'Inter', sans-serif;
      font-size: 13px;
      padding: 9px 12px;
      outline: none;
      transition: border-color .2s;
    }
    .llm-select { cursor: pointer; }
    .llm-select:focus, .llm-input:focus { border-color: var(--purple); }
    .llm-select option { background: var(--surface); }

    #result-section  { display: none; }
    #rag-result-section { display: none; }

    /* ── Regulation Timeline ── */
    .timeline-wrapper {
      position: relative;
      padding-left: 32px;
    }
    .timeline-wrapper::before {
      content: '';
      position: absolute;
      left: 10px;
      top: 8px;
      bottom: 8px;
      width: 2px;
      background: linear-gradient(180deg, var(--accent) 0%, var(--accent2) 50%, var(--warn) 100%);
      opacity: .45;
      border-radius: 2px;
    }
    .tl-item {
      position: relative;
      margin-bottom: 22px;
    }
    .tl-dot {
      position: absolute;
      left: -26px;
      top: 10px;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: var(--bg);
      border: 2px solid var(--accent);
      box-shadow: 0 0 6px rgba(88,166,255,.4);
      z-index: 1;
    }
    .tl-dot.current {
      border-color: var(--accent2);
      box-shadow: 0 0 8px rgba(63,185,80,.6);
      background: rgba(63,185,80,.15);
    }
    .tl-dot.old {
      border-color: var(--muted);
      box-shadow: none;
    }
    .tl-card {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      transition: border-color .2s;
    }
    .tl-card:hover { border-color: var(--accent); }
    .tl-card.current { border-color: rgba(63,185,80,.5); background: rgba(63,185,80,.04); }
    .tl-card.old     { opacity: .75; }
    .tl-header {
      display: flex; align-items: center; gap: 8px;
      margin-bottom: 10px; flex-wrap: wrap;
    }
    .tl-version {
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px; font-weight: 600;
      color: var(--accent);
    }
    .tl-date {
      font-size: 11px; color: var(--muted);
      font-family: 'JetBrains Mono', monospace;
    }
    .tl-excerpt {
      font-size: 13px; line-height: 1.65;
      color: var(--text); white-space: pre-wrap; word-break: break-word;
    }

    /* ── Legal Evidence Cards ── */
    .evidence-card {
      background: var(--bg);
      border: 1px solid rgba(210,153,34,.35);
      border-radius: 8px;
      padding: 14px 16px;
      margin-bottom: 10px;
    }
    .evidence-meta {
      display: flex; flex-wrap: wrap; gap: 6px;
      margin-bottom: 8px; align-items: center;
    }
    .evidence-excerpt {
      font-size: 13px; line-height: 1.6;
      color: var(--muted); font-style: italic;
      border-left: 3px solid rgba(210,153,34,.4);
      padding-left: 10px; margin-top: 6px;
    }
  </style>
</head>
<body>

<header>
  <div>
    <div class="logo">TDDR</div>
    <div class="logo-sub">Temporal Regulation Drift Detector</div>
  </div>
    <div class="header-right">
    <div class="badge">Module 1 &nbsp;&#x2713;</div>
    <div class="badge green">Module 2 RAG &nbsp;&#x2713;</div>
    <div class="badge" style="color:var(--purple);background:rgba(188,140,255,.1);border-color:rgba(188,140,255,.25);">Module 3 LLM &nbsp;&#x2713;</div>
    <div class="badge" style="color:var(--warn);background:rgba(210,153,34,.1);border-color:rgba(210,153,34,.25);">Module 4 Output &nbsp;&#x2713;</div>
  </div>
</header>

<!-- Tabs -->
<div class="tabs">
  <button class="tab-btn active" id="tab-query" onclick="switchTab('query')">
    <span class="dot"></span> Query Parser
  </button>
  <button class="tab-btn" id="tab-rag" onclick="switchTab('rag')">
    <span class="dot"></span> RAG Retrieval
  </button>
  <button class="tab-btn" id="tab-llm" onclick="switchTab('llm')">
    <span class="dot" style="background:var(--purple);"></span> LLM Answer
  </button>
  <button class="tab-btn" id="tab-timeline" onclick="switchTab('timeline')">
    <span class="dot" style="background:var(--warn);"></span> Regulation Timeline
  </button>
  <button class="tab-btn" id="tab-fullpipeline" onclick="switchTab('fullpipeline')">
    <span class="dot" style="background:var(--accent2);"></span> Full Pipeline Flow
  </button>
</div>

<main>

  <!-- ══════════════════════════════════════════════════════ TAB 1: Query Parser -->
  <div class="tab-panel active" id="panel-query">

    <div class="card">
      <div class="card-label">Legal Query</div>
      <textarea id="query" class="query-area"
        placeholder="e.g.  What was the punishment under Section 302 IPC before 2018?"
        rows="3"></textarea>

      <div class="examples">
        <button class="example-chip" onclick="useExample(this,'query')">Section 21 IPC before 2018</button>
        <button class="example-chip" onclick="useExample(this,'query')">Compare Section 21 and Section 22 IPC before 2018</button>
        <button class="example-chip" onclick="useExample(this,'query')">Define Section 375 IPC as of June 2015</button>
        <button class="example-chip" onclick="useExample(this,'query')">Was Section 124A IPC valid between 2010 and 2020?</button>
        <button class="example-chip" onclick="useExample(this,'query')">Section 66A IT Act amended after 2015</button>
        <button class="example-chip" onclick="useExample(this,'query')">What does Section 21A say?</button>
      </div>

      <div class="btn-row">
        <button class="btn-submit" id="runBtn" onclick="runQuery()">Parse Query</button>
        <button class="btn-clear" onclick="clearQuery()">Clear</button>
        <div class="spinner" id="spinner"></div>
        <span class="hint">Ctrl + Enter to run</span>
      </div>
    </div>

    <!-- Result -->
    <div id="result-section">
      <div class="card">
        <div class="result-header">
          <div class="card-label" style="margin:0">Pipeline Result</div>
          <div class="pipeline-tags">
            <span class="tag active">Layer A &nbsp;✓</span>
            <span class="tag active">Layer B &nbsp;✓</span>
            <span class="tag active">Layer B.5 &nbsp;✓</span>
            <span class="tag active">Layer C &nbsp;✓</span>
          </div>
        </div>
        <div class="summary-grid" id="summary-grid"></div>
      </div>

      <div class="card">
        <div class="result-header">
          <div class="card-label" style="margin:0">Structured Query · JSON</div>
          <button class="copy-btn" id="copy-btn-q" onclick="copyJson('q')">Copy JSON</button>
        </div>
        <div class="json-block" id="json-out"></div>
      </div>
    </div>

    <div id="error-section" style="display:none">
      <div class="error-msg" id="error-msg"></div>
    </div>

  </div><!-- /panel-query -->


  <!-- ══════════════════════════════════════════════════════ TAB 2: Full Pipeline -->
  <div class="tab-panel" id="panel-rag">

    <!-- Store status -->
    <div class="store-bar">
      <div class="store-dot" id="store-dot" class="loaded"></div>
      <span id="store-status">Vector store: loading IPC law data…</span>
    </div>

    <!-- Pipeline query input -->
    <div class="card">
      <div class="card-label">Full Pipeline · Legal Query → Module 1 → Module 2</div>
      <p style="font-size:12px;color:var(--muted);margin-bottom:12px;">
        Enter a natural-language legal query. It is parsed by <strong>Module 1</strong> (user_input_module)
        to extract entities, intent, and temporal constraints, then passed to <strong>Module 2</strong>
        (rag_module) for temporally-filtered vector retrieval. No manual date entry needed.
      </p>
      <textarea id="rag-query" class="query-area" rows="3"
        placeholder="e.g.  What was the punishment under Section 302 IPC before 2018?"></textarea>

      <div class="examples" style="margin-top:12px;">
        <button class="example-chip" onclick="useRagExample('What was the punishment under Section 302 IPC before 2018?')">Section 302 before 2018</button>
        <button class="example-chip" onclick="useRagExample('What is the current punishment for Section 302 IPC?')">Section 302 current</button>
        <button class="example-chip" onclick="useRagExample('Define Section 375 IPC as of June 2015')">Section 375 as of 2015</button>
        <button class="example-chip" onclick="useRagExample('Was Section 124A IPC valid between 2010 and 2020?')">Section 124A 2010–2020</button>
        <button class="example-chip" onclick="useRagExample('How was Section 66A IT Act amended after 2015?')">Section 66A after 2015</button>
        <button class="example-chip" onclick="useRagExample('What does Section 21 IPC say?')">Section 21 (no date)</button>
      </div>

      <div class="btn-row">
        <button class="btn-submit purple" id="retrieveBtn" onclick="runPipeline()">Run Full Pipeline</button>
        <button class="btn-clear" onclick="clearPipeline()">Clear</button>
        <div class="spinner" id="retrieve-spinner"></div>
        <span class="hint">Module 1 → Module 2</span>
      </div>
    </div>

    <!-- Module 1 summary -->
    <div id="pipeline-m1-section" style="display:none">
      <div class="card">
        <div class="result-header">
          <div class="card-label" style="margin:0">Module 1 · Structured Query</div>
          <div class="pipeline-tags">
            <span class="tag active">Layer A ✓</span>
            <span class="tag active">Layer B ✓</span>
            <span class="tag active">Layer B.5 ✓</span>
            <span class="tag active">Layer C ✓</span>
          </div>
        </div>
        <div class="summary-grid" id="pipeline-summary-grid"></div>
      </div>
    </div>

    <!-- Module 2 results -->
    <div id="rag-result-section" style="display:none">
      <div class="card">
        <div class="result-header">
          <div class="card-label" style="margin:0">Module 2 · Retrieved Clauses</div>
          <div class="pipeline-tags" id="rag-pipeline-tags">
            <span class="tag purple">Temporal Filter ✓</span>
            <span class="tag purple">Vector Search ✓</span>
          </div>
        </div>
        <div class="chunk-list" id="chunk-list"></div>
      </div>
    </div>

    <div id="rag-error-section" style="display:none">
      <div class="error-msg" id="rag-error-msg"></div>
    </div>

  </div><!-- /panel-rag -->


  <!-- ══════════════════════════════════════════════════════ TAB 3: Full Pipeline - LLM Answer -->
  <div class="tab-panel" id="panel-llm">

    <!-- Store status bar (reused) -->
    <div class="store-bar" id="llm-store-bar">
      <div class="store-dot" id="llm-store-dot"></div>
      <span id="llm-store-status">Vector store: checking…</span>
    </div>

    <!-- Pipeline architecture diagram -->
    <div class="card" style="padding:18px 22px;">
      <div class="card-label">Full Pipeline Architecture</div>
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:2px;">
        <div style="text-align:center;">
          <div class="tag" style="font-size:12px;padding:5px 12px;color:var(--text);">💬 User Query</div>
        </div>
        <span style="color:var(--muted);">→</span>
        <div style="text-align:center;">
          <div class="tag active" style="font-size:12px;padding:5px 12px;">Module 1</div>
          <div style="font-size:10px;color:var(--muted);margin-top:3px;">user_input_module</div>
        </div>
        <span style="color:var(--muted);">→</span>
        <div style="text-align:center;">
          <div class="tag" style="font-size:12px;padding:5px 12px;color:var(--accent2);background:rgba(63,185,80,.1);border-color:rgba(63,185,80,.3);">Module 2</div>
          <div style="font-size:10px;color:var(--muted);margin-top:3px;">rag_module</div>
        </div>
        <span style="color:var(--muted);">→</span>
        <div style="text-align:center;">
          <div class="tag" style="font-size:12px;padding:5px 12px;color:var(--purple);background:rgba(188,140,255,.1);border-color:rgba(188,140,255,.3);">Module 3</div>
          <div style="font-size:10px;color:var(--muted);margin-top:3px;">llm_module</div>
        </div>
        <span style="color:var(--muted);">→</span>
        <div style="text-align:center;">
          <div class="tag" style="font-size:12px;padding:5px 12px;color:var(--warn);background:rgba(210,153,34,.1);border-color:rgba(210,153,34,.3);">✨ Answer</div>
        </div>
      </div>
    </div>

    <!-- Query input -->
    <div class="card">
      <div class="card-label">Legal Query</div>
      <textarea id="llm-query" class="query-area" rows="3"
        placeholder="e.g.  What was the punishment under Section 302 IPC before 2018?"></textarea>

      <div class="examples" style="margin-top:10px;">
        <button class="example-chip" onclick="useLlmExample('What was the punishment under Section 302 IPC before 2018?')">Section 302 before 2018</button>
        <button class="example-chip" onclick="useLlmExample('Define Section 375 IPC as of June 2015')">Section 375 as of 2015</button>
        <button class="example-chip" onclick="useLlmExample('Was Section 66A IT Act valid after 2015?')">Section 66A after 2015</button>
        <button class="example-chip" onclick="useLlmExample('What is the current punishment for murder under IPC?')">Murder punishment (current)</button>
        <button class="example-chip" onclick="useLlmExample('Explain Section 124A sedition law between 2010 and 2020')">Sedition 2010–2020</button>
      </div>

      <!-- LLM config -->
      <div style="margin-top:14px;padding-top:14px;border-top:1px solid var(--border);">
        <div class="card-label" style="margin-bottom:10px;">LLM Backend Configuration</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;align-items:end;">
          <div>
            <div class="field-label">Backend</div>
            <select id="llm-backend" class="llm-select" onchange="onBackendChange()">
              <option value="mock">🧪 Mock (no setup)</option>
              <option value="ollama">💻 Ollama (local)</option>
              <option value="openai">🌐 OpenAI</option>
              <option value="huggingface">🤗 HuggingFace</option>
            </select>
          </div>
          <div>
            <div class="field-label">Model</div>
            <input id="llm-model" type="text" class="llm-input" placeholder="auto" />
          </div>
          <div>
            <div class="field-label">API Key (if needed)</div>
            <input id="llm-apikey" type="password" class="llm-input" placeholder="sk-... or hf_..."/>
          </div>
        </div>
        <div id="llm-backend-hint" style="margin-top:8px;font-size:11px;color:var(--muted);">
          Mock backend requires no setup — ideal for demos and testing.
        </div>
      </div>

      <div class="btn-row" style="margin-top:14px;">
        <button class="btn-submit" style="background:var(--purple);" id="llmBtn" onclick="runLLMPipeline()">✨ Generate Answer</button>
        <button class="btn-clear" onclick="clearLLM()">Clear</button>
        <div class="spinner" id="llm-spinner"></div>
        <span class="hint">Module 1 → Module 2 → Module 3</span>
      </div>
    </div>

    <!-- Pipeline progress indicator -->
    <div id="llm-pipeline-progress" style="display:none;">
      <div style="display:flex;align-items:center;gap:8px;padding:12px 18px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);">
        <span style="font-size:12px;color:var(--muted);font-family:'JetBrains Mono',monospace;">Pipeline:</span>
        <span class="tag" id="prog-m1">M1: Parsing</span>
        <span style="color:var(--muted);">→</span>
        <span class="tag" id="prog-m2">M2: Retrieving</span>
        <span style="color:var(--muted);">→</span>
        <span class="tag" id="prog-m3">M3: Generating</span>
      </div>
    </div>

    <!-- Module 1 structured query summary -->
    <div id="llm-m1-section" style="display:none;">
      <div class="card">
        <div class="result-header">
          <div class="card-label" style="margin:0">Module 1 — Structured Query</div>
          <div class="pipeline-tags">
            <span class="tag active">Layer A ✓</span>
            <span class="tag active">Layer B ✓</span>
            <span class="tag active">Layer B.5 ✓</span>
            <span class="tag active">Layer C ✓</span>
          </div>
        </div>
        <div class="summary-grid" id="llm-m1-summary"></div>
      </div>
    </div>

    <!-- Module 2 retrieved chunks (collapsed) -->
    <div id="llm-m2-section" style="display:none;">
      <div class="card">
        <div class="result-header">
          <div class="card-label" style="margin:0">Module 2 — Retrieved Regulation Chunks</div>
          <div class="pipeline-tags" id="llm-m2-tags"></div>
        </div>
        <div class="chunk-list" id="llm-chunk-list"></div>
      </div>
    </div>

    <!-- Module 3 LLM Answer -->
    <div id="llm-m3-section" style="display:none;">

      <!-- Main answer card -->
      <div class="card" style="border-color:rgba(188,140,255,.35);background:linear-gradient(135deg,rgba(188,140,255,.05),rgba(63,185,80,.03));">
        <div class="result-header">
          <div class="card-label" style="margin:0;color:var(--purple);">✨ Module 3 — LLM Generated Answer</div>
          <div class="pipeline-tags" id="llm-m3-tags"></div>
        </div>

        <div id="llm-answer-text" style="
          font-size:15px;line-height:1.75;color:var(--text);
          padding:16px;background:var(--bg);border:1px solid var(--border);
          border-radius:8px;margin-bottom:14px;white-space:pre-wrap;
        "></div>

        <!-- Citations -->
        <div id="llm-citations-row" style="display:none;margin-bottom:14px;">
          <div class="card-label">Cited Sections</div>
          <div id="llm-citations" style="display:flex;flex-wrap:wrap;gap:7px;"></div>
        </div>

        <!-- Explanation -->
        <div id="llm-explanation-row" style="display:none;">
          <div class="card-label">Legal Basis</div>
          <div id="llm-explanation" style="
            font-size:13px;line-height:1.65;color:var(--muted);
            padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:8px;
          "></div>
        </div>
      </div>

    </div><!-- /llm-m3-section -->

    <div id="llm-error-section" style="display:none;">
      <div class="error-msg" id="llm-error-msg"></div>
    </div>

  </div><!-- /panel-llm -->


  <!-- ══════════════════════════════════════════════════════ TAB 4: Regulation Timeline -->
  <div class="tab-panel" id="panel-timeline">

    <div class="card">
      <div class="card-label">Regulation Timeline Visualization — Module 4 Output</div>
      <p style="font-size:12px;color:var(--muted);margin-bottom:12px;">
        Runs the full M1→M2→M3→M4 pipeline and visualises
        <strong>all temporal versions</strong> of every cited regulation as a vertical timeline.
        Best results with SWM Rules queries.
      </p>
      <textarea id="tl-query" class="query-area" rows="2"
        placeholder="e.g. What are the solid waste disposal rules in India?"></textarea>

      <div class="examples" style="margin-top:10px;">
        <button class="example-chip" onclick="useTlExample('What are the solid waste disposal rules under SWM Rules?')">SWM Rules disposal</button>
        <button class="example-chip" onclick="useTlExample('What is the duty of local bodies under solid waste management rules?')">Local body duties (SWM)</button>
        <button class="example-chip" onclick="useTlExample('Explain Rule 5 of Solid Waste Management Rules')">Rule 5 SWM</button>
        <button class="example-chip" onclick="useTlExample('When did extended producer responsibility apply for solid waste?')">EPR solid waste</button>
        <button class="example-chip" onclick="useTlExample('How was Section 302 IPC punishment defined before 2018?')">IPC Section 302</button>
      </div>

      <div class="btn-row" style="margin-top:14px;">
        <button class="btn-submit" style="background:var(--warn);color:#0d1117;" id="tlBtn" onclick="runTimeline()">⏱ Build Timeline</button>
        <button class="btn-clear" onclick="clearTimeline()">Clear</button>
        <div class="spinner" id="tl-spinner"></div>
        <span class="hint">M1 → M2 → M3 → M4 → Timeline</span>
      </div>
    </div>

    <!-- Answer summary -->
    <div id="tl-answer-section" style="display:none;">
      <div class="card" style="border-color:rgba(210,153,34,.35);background:linear-gradient(135deg,rgba(210,153,34,.06),rgba(63,185,80,.03));">
        <div class="result-header">
          <div class="card-label" style="margin:0;color:var(--warn);">✨ Answer</div>
          <div id="tl-conf-tags" class="pipeline-tags"></div>
        </div>
        <div id="tl-answer-text" style="font-size:15px;line-height:1.75;color:var(--text);padding:14px;background:var(--bg);border:1px solid var(--border);border-radius:8px;white-space:pre-wrap;"></div>
      </div>
    </div>

    <!-- Timeline cards -->
    <div id="tl-timeline-section" style="display:none;">
      <div class="card">
        <div class="result-header">
          <div class="card-label" style="margin:0;">📅 Regulation Version Timeline</div>
          <div id="tl-timeline-tags" class="pipeline-tags"></div>
        </div>
        <div id="tl-timeline-container" style="margin-top:16px;"></div>
      </div>
    </div>

    <div id="tl-error-section" style="display:none;">
      <div class="error-msg" id="tl-error-msg"></div>
    </div>

  </div><!-- /panel-timeline -->


  <!-- ══════════════════════════════════════════════════════ TAB 5: Full Pipeline Flow -->
  <div class="tab-panel" id="panel-fullpipeline">

    <!-- Architecture diagram (animated after run) -->
    <div class="card" style="padding:18px 22px;" id="fp-arch-card">
      <div class="card-label">Complete TDDR Pipeline — All 4 Modules</div>
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-top:6px;">
        <div style="text-align:center;">
          <div class="tag" style="font-size:12px;padding:5px 12px;color:var(--text);">💬 User Query</div>
        </div>
        <span style="color:var(--muted);">→</span>
        <div style="text-align:center;">
          <div class="tag active" id="fp-m1-arch" style="font-size:12px;padding:5px 12px;">Module 1</div>
          <div style="font-size:10px;color:var(--muted);margin-top:3px;">user_input_module</div>
        </div>
        <span style="color:var(--muted);">→</span>
        <div style="text-align:center;">
          <div class="tag" id="fp-m2-arch" style="font-size:12px;padding:5px 12px;color:var(--accent2);background:rgba(63,185,80,.1);border-color:rgba(63,185,80,.3);">Module 2</div>
          <div style="font-size:10px;color:var(--muted);margin-top:3px;">rag_module</div>
        </div>
        <span style="color:var(--muted);">→</span>
        <div style="text-align:center;">
          <div class="tag" id="fp-m3-arch" style="font-size:12px;padding:5px 12px;color:var(--purple);background:rgba(188,140,255,.1);border-color:rgba(188,140,255,.3);">Module 3</div>
          <div style="font-size:10px;color:var(--muted);margin-top:3px;">llm_module</div>
        </div>
        <span style="color:var(--muted);">→</span>
        <div style="text-align:center;">
          <div class="tag" id="fp-m4-arch" style="font-size:12px;padding:5px 12px;color:var(--warn);background:rgba(210,153,34,.1);border-color:rgba(210,153,34,.3);">Module 4</div>
          <div style="font-size:10px;color:var(--muted);margin-top:3px;">output_module</div>
        </div>
        <span style="color:var(--muted);">→</span>
        <div style="text-align:center;">
          <div class="tag" style="font-size:12px;padding:5px 12px;color:var(--accent2);background:rgba(63,185,80,.08);border-color:rgba(63,185,80,.25);">🏁 Final Response</div>
        </div>
      </div>
    </div>

    <!-- Input -->
    <div class="card">
      <div class="card-label">Legal Query</div>
      <textarea id="fp-query" class="query-area" rows="3"
        placeholder="e.g. Explain Rule 5 of Solid Waste Management Rules 2016"></textarea>

      <div class="examples" style="margin-top:10px;">
        <button class="example-chip" onclick="useFpExample('Explain Rule 5 of Solid Waste Management Rules 2016')">SWM Rule 5 (2016)</button>
        <button class="example-chip" onclick="useFpExample('What is the duty of waste generators under SWM Rules?')">Waste generator duties</button>
        <button class="example-chip" onclick="useFpExample('Was Section 66A IT Act valid after 2015?')">Section 66A after 2015</button>
        <button class="example-chip" onclick="useFpExample('What was the punishment under Section 302 IPC before 2018?')">Section 302 before 2018</button>
        <button class="example-chip" onclick="useFpExample('What is Extended Producer Responsibility under SWM Rules?')">EPR under SWM Rules</button>
      </div>

      <!-- LLM config -->
      <div style="margin-top:14px;padding-top:14px;border-top:1px solid var(--border);">
        <div class="card-label" style="margin-bottom:10px;">LLM Backend</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;align-items:end;">
          <div>
            <div class="field-label">Backend</div>
            <select id="fp-backend" class="llm-select" onchange="onFpBackendChange()">
              <option value="mock">🧪 Mock (no setup)</option>
              <option value="ollama">💻 Ollama (local)</option>
              <option value="openai">🌐 OpenAI</option>
              <option value="huggingface">🤗 HuggingFace</option>
            </select>
          </div>
          <div>
            <div class="field-label">Model</div>
            <input id="fp-model" type="text" class="llm-input" placeholder="auto" />
          </div>
          <div>
            <div class="field-label">API Key (if needed)</div>
            <input id="fp-apikey" type="password" class="llm-input" placeholder="sk-... or hf_..."/>
          </div>
        </div>
      </div>

      <div class="btn-row" style="margin-top:14px;">
        <button class="btn-submit green" id="fpBtn" onclick="runFullPipeline()">🔁 Run Full Pipeline</button>
        <button class="btn-clear" onclick="clearFullPipeline()">Clear</button>
        <div class="spinner" id="fp-spinner"></div>
        <span class="hint">Module 1 → 2 → 3 → 4</span>
      </div>
    </div>

    <!-- Pipeline progress -->
    <div id="fp-pipeline-progress" style="display:none;">
      <div style="display:flex;align-items:center;gap:8px;padding:12px 18px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);">
        <span style="font-size:12px;color:var(--muted);font-family:'JetBrains Mono',monospace;">Pipeline:</span>
        <span class="tag" id="fp-prog-m1">M1: Parsing</span>
        <span style="color:var(--muted);">→</span>
        <span class="tag" id="fp-prog-m2">M2: Retrieving</span>
        <span style="color:var(--muted);">→</span>
        <span class="tag" id="fp-prog-m3">M3: Generating</span>
        <span style="color:var(--muted);">→</span>
        <span class="tag" id="fp-prog-m4">M4: Building</span>
      </div>
    </div>

    <!-- Module 1 -->
    <div id="fp-m1-section" style="display:none;">
      <div class="card">
        <div class="result-header">
          <div class="card-label" style="margin:0;">Module 1 — Structured Query</div>
          <div class="pipeline-tags"><span class="tag active">Layer A ✓</span><span class="tag active">Layer B ✓</span><span class="tag active">Layer C ✓</span></div>
        </div>
        <div class="summary-grid" id="fp-m1-grid"></div>
      </div>
    </div>

    <!-- Module 2 -->
    <div id="fp-m2-section" style="display:none;">
      <div class="card">
        <div class="result-header">
          <div class="card-label" style="margin:0;">Module 2 — Retrieved Chunks</div>
          <div class="pipeline-tags" id="fp-m2-tags"></div>
        </div>
        <div class="chunk-list" id="fp-chunk-list"></div>
      </div>
    </div>

    <!-- Module 3 -->
    <div id="fp-m3-section" style="display:none;">
      <div class="card" style="border-color:rgba(188,140,255,.35);">
        <div class="result-header">
          <div class="card-label" style="margin:0;color:var(--purple);">Module 3 — LLM Answer</div>
          <div class="pipeline-tags" id="fp-m3-tags"></div>
        </div>
        <div id="fp-answer-text" style="font-size:15px;line-height:1.75;color:var(--text);padding:14px;background:var(--bg);border:1px solid var(--border);border-radius:8px;white-space:pre-wrap;margin-bottom:12px;"></div>
        <div id="fp-citations-row" style="display:none;margin-bottom:10px;">
          <div class="card-label">Cited Sections</div>
          <div id="fp-citations" style="display:flex;flex-wrap:wrap;gap:7px;"></div>
        </div>
      </div>
    </div>

    <!-- Module 4 — FinalSystemResponse -->
    <div id="fp-m4-section" style="display:none;">
      <div class="card" style="border-color:rgba(210,153,34,.45);background:linear-gradient(135deg,rgba(210,153,34,.06),rgba(88,166,255,.03));">
        <div class="result-header">
          <div class="card-label" style="margin:0;color:var(--warn);">🏁 Module 4 — Final System Response</div>
          <div class="pipeline-tags" id="fp-m4-tags"></div>
        </div>

        <!-- Legal Basis -->
        <div style="margin-bottom:14px;">
          <div class="card-label">Legal Basis (Resolved Citations)</div>
          <div id="fp-legal-basis"></div>
        </div>

        <!-- Explanation -->
        <div id="fp-explanation-row" style="display:none;margin-bottom:14px;">
          <div class="card-label">Explanation</div>
          <div id="fp-explanation" style="font-size:13px;line-height:1.65;color:var(--muted);padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:8px;"></div>
        </div>

        <!-- Raw JSON -->
        <div>
          <div class="result-header" style="margin-bottom:10px;">
            <div class="card-label" style="margin:0;">FinalSystemResponse · JSON</div>
            <button class="copy-btn" id="fp-copy-btn" onclick="copyFpJson()">Copy JSON</button>
          </div>
          <div class="json-block" id="fp-json-out"></div>
        </div>
      </div>
    </div>

    <div id="fp-error-section" style="display:none;">
      <div class="error-msg" id="fp-error-msg"></div>
    </div>

  </div><!-- /panel-fullpipeline -->


</main>

<script>
  // ── Tab switching ────────────────────────────────────────────────────────────
  function switchTab(name) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('panel-' + name).classList.add('active');
    document.getElementById('tab-' + name).classList.add('active');
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  const qa = document.getElementById('query');
  qa.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runQuery();
  });

  function useExample(btn, field) {
    document.getElementById(field === 'query' ? 'query' : field).value = btn.textContent.trim();
  }

  function useRagExample(q) {
    document.getElementById('rag-query').value = q;
  }

  // ── Example documents ─────────────────────────────────────────────────────
  const EXAMPLES = {
    osha2017: {
      regulation: "OSHA", version: "2017",
      effective_from: "2017-01-01", effective_to: "2020-12-31",
      text: "Section 1 Chemical Storage\\nChemicals must be stored in ventilated containers away from heat sources. All storage areas must be labelled with appropriate hazard signs.\\n\\nSection 2 Handling Procedures\\nEmployees must wear protective equipment when handling corrosive materials. Gloves, goggles, and aprons are mandatory.\\n\\nSection 3 Disposal\\nChemical waste must be collected in designated containers. Disposal must comply with local environmental regulations."
    },
    osha2021: {
      regulation: "OSHA", version: "2021",
      effective_from: "2021-01-01", effective_to: null,
      text: "Section 1 Chemical Storage\\nAll storage facilities must be equipped with automated ventilation systems and continuous air quality monitoring. Containers must display QR-coded hazard data sheets.\\n\\nSection 2 Personal Protective Equipment\\nEmployees must pass annual PPE certification. Full chemical suits are required for Class-A hazardous materials.\\n\\nSection 3 Disposal\\nAll chemical waste must be tracked via a digital chain-of-custody system before disposal."
    },
    gdpr2018: {
      regulation: "GDPR", version: "2018",
      effective_from: "2018-05-25", effective_to: null,
      text: "Article 5 Principles\\nPersonal data must be processed lawfully, fairly, and transparently. Data collected for specified, explicit, and legitimate purposes must not be processed further.\\n\\nArticle 6 Lawfulness\\nProcessing is lawful only if the data subject has given consent, or if processing is necessary for the performance of a contract.\\n\\nArticle 17 Right to Erasure\\nThe data subject has the right to obtain from the controller the erasure of personal data concerning him or her without undue delay."
    }
  };

  function loadExample(key) {
    document.getElementById('doc-input').value = JSON.stringify(EXAMPLES[key], null, 2);
  }

  // ── Tab 1: Query Parser ───────────────────────────────────────────────────
  let _lastQueryJson = '';

  function clearQuery() {
    qa.value = '';
    document.getElementById('result-section').style.display = 'none';
    document.getElementById('error-section').style.display  = 'none';
    qa.focus();
  }

  async function runQuery() {
    const q = qa.value.trim();
    if (!q) { qa.focus(); return; }

    document.getElementById('result-section').style.display = 'none';
    document.getElementById('error-section').style.display  = 'none';
    document.getElementById('runBtn').disabled = true;
    document.getElementById('spinner').style.display = 'inline-block';

    try {
      const res  = await fetch('/api/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q }),
      });
      const data = await res.json();
      if (data.error) showError(data.error);
      else            showResult(data);
    } catch (err) {
      showError('Network error: ' + err.message);
    } finally {
      document.getElementById('runBtn').disabled = false;
      document.getElementById('spinner').style.display = 'none';
    }
  }

  function showError(msg) {
    document.getElementById('error-msg').textContent = msg;
    document.getElementById('error-section').style.display = 'block';
  }

  function showResult(data) {
    _lastQueryJson = JSON.stringify(data, null, 2);
    const f = data.filters || {};
    const items = [
      { label: 'Semantic Query',    value: data.semantic_query,                cls: '' },
      { label: 'Canonical ID',      value: data.canonical_entity_id || '—',    cls: data.canonical_entity_id ? 'highlight' : 'grey' },
      { label: 'Entity Type',       value: data.entity_type || '—',            cls: data.entity_type ? '' : 'grey' },
      { label: 'Act',               value: f.act_name || '—',                  cls: f.act_name ? 'green' : 'grey' },
      { label: 'Section (primary)', value: f.section_id || '—',                cls: f.section_id ? '' : 'grey' },
      { label: 'All Sections',      value: (f.section_ids && f.section_ids.length) ? f.section_ids.join(', ') : '—', cls: '' },
      { label: 'Temporal Operator', value: (f.valid_time || {}).operator || '—',        cls: '' },
      { label: 'Reference Date',    value: (f.valid_time || {}).reference_date || '—',  cls: '' },
    ];
    if (f.canonical_entity_ids && f.canonical_entity_ids.length > 1)
      items.push({ label: 'All Canonical IDs', value: f.canonical_entity_ids.join(' · '), cls: 'highlight' });

    document.getElementById('summary-grid').innerHTML = items.map(it => `
      <div class="summary-item">
        <div class="s-label">${it.label}</div>
        <div class="s-value ${it.cls}">${it.value}</div>
      </div>`).join('');

    document.getElementById('json-out').innerHTML = colorJson(_lastQueryJson);
    document.getElementById('result-section').style.display = 'block';
  }

  function copyJson(which) {
    const text = which === 'q' ? _lastQueryJson : (_lastRagJson || '');
    const btnId = which === 'q' ? 'copy-btn-q' : 'copy-btn-rag';
    navigator.clipboard.writeText(text).then(() => {
      const b = document.getElementById(btnId);
      b.textContent = 'Copied!';
      setTimeout(() => b.textContent = 'Copy JSON', 1500);
    });
  }

  // ── Tab 2: Full Pipeline ─────────────────────────────────────────────────

  // Poll store status on page load
  (async function initStoreStatus() {
    try {
      const res  = await fetch('/api/rag/status', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: '{}' });
      const data = await res.json();
      updateStoreStatus(data.total_chunks || 0, data.rag_ready);
    } catch(e) { /* ignore */ }
  })();

  function updateStoreStatus(count, ready) {
    const dot    = document.getElementById('store-dot');
    const status = document.getElementById('store-status');
    if (!ready) {
      dot.classList.remove('loaded');
      status.textContent = 'RAG module unavailable — install dependencies (faiss-cpu, sentence-transformers).';
    } else if (count === 0) {
      dot.classList.remove('loaded');
      status.textContent = 'Vector store: empty — still loading…';
    } else {
      dot.classList.add('loaded');
      status.textContent = `Vector store ready · ${count} chunk${count===1?'':'s'} (IPC + IT Act pre-loaded).`;
    }
  }

  let _lastRagJson = '';

  function clearPipeline() {
    document.getElementById('rag-query').value = '';
    document.getElementById('rag-result-section').style.display    = 'none';
    document.getElementById('pipeline-m1-section').style.display   = 'none';
    document.getElementById('rag-error-section').style.display     = 'none';
  }

  async function runPipeline() {
    const q = document.getElementById('rag-query').value.trim();
    if (!q) { document.getElementById('rag-query').focus(); return; }

    document.getElementById('rag-result-section').style.display    = 'none';
    document.getElementById('pipeline-m1-section').style.display   = 'none';
    document.getElementById('rag-error-section').style.display     = 'none';
    document.getElementById('retrieveBtn').disabled = true;
    document.getElementById('retrieve-spinner').style.display = 'inline-block';

    try {
      const res  = await fetch('/api/rag/pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, k: 5 }),
      });
      const data = await res.json();
      if (data.error) showRagError(data.error);
      else            showPipelineResult(data);
    } catch (err) {
      showRagError('Network error: ' + err.message);
    } finally {
      document.getElementById('retrieveBtn').disabled = false;
      document.getElementById('retrieve-spinner').style.display = 'none';
    }
  }

  function showPipelineResult(data) {
    _lastRagJson = JSON.stringify(data, null, 2);
    const sq = data.structured_query || {};
    const f  = sq.filters || {};
    const vt = f.valid_time || {};

    // Module 1 summary
    const items = [
      { label: 'Semantic Query',    value: sq.semantic_query || '—',    cls: '' },
      { label: 'Canonical ID',      value: sq.canonical_entity_id || '—', cls: sq.canonical_entity_id ? 'highlight' : 'grey' },
      { label: 'Entity Type',       value: sq.entity_type || '—',       cls: sq.entity_type ? '' : 'grey' },
      { label: 'Act',               value: f.act_name || '—',           cls: f.act_name ? 'green' : 'grey' },
      { label: 'Section (primary)', value: f.section_id || '—',         cls: f.section_id ? '' : 'grey' },
      { label: 'Temporal Operator', value: vt.operator || '—',          cls: '' },
      { label: 'Reference Date',    value: vt.reference_date || '—',    cls: vt.reference_date ? 'highlight' : 'grey' },
    ];
    document.getElementById('pipeline-summary-grid').innerHTML = items.map(it => `
      <div class="summary-item">
        <div class="s-label">${it.label}</div>
        <div class="s-value ${it.cls}">${it.value}</div>
      </div>`).join('');
    document.getElementById('pipeline-m1-section').style.display = 'block';

    // Module 2 chunks
    showRagResult({ query_date: vt.reference_date, chunks: data.chunks || [] });
  }

  function showRagError(msg) {
    document.getElementById('rag-error-msg').textContent = msg;
    document.getElementById('rag-error-section').style.display = 'block';
  }

  function showRagResult(data) {
    _lastRagJson = JSON.stringify(data, null, 2);
    const chunks = data.chunks || [];

    const tags = document.getElementById('rag-pipeline-tags');
    const dateLabel = data.query_date
      ? `Temporal Filter · ${data.query_date} ✓`
      : 'Temporal Filter · all time ✓';
    tags.innerHTML = `
      <span class="tag purple">${dateLabel}</span>
      <span class="tag purple">Vector Search ✓</span>
      <span class="tag ok">${chunks.length} result${chunks.length===1?'':'s'}</span>`;

    if (chunks.length === 0) {
      document.getElementById('chunk-list').innerHTML = `
        <div style="color:var(--muted);font-size:13px;padding:8px 0;">
          No chunks matched the query for the given time period.
        </div>`;
    } else {
      document.getElementById('chunk-list').innerHTML = chunks.map((c, i) => `
        <div class="chunk-card">
          <div class="chunk-meta">
            <div class="chunk-rank">${i+1}</div>
            <span class="tag purple">${c.regulation}</span>
            <span class="tag ok">v${c.version}</span>
            <span class="tag">${c.effective_from}${c.effective_to ? ' → '+c.effective_to : ' → now'}</span>
            <span class="tag" style="margin-left:auto;font-size:10px;color:var(--muted);">${c.chunk_id}</span>
          </div>
          <div class="chunk-text">${escHtml(c.text)}</div>
        </div>`).join('');
    }

    document.getElementById('rag-result-section').style.display = 'block';
  }

  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  // ── Tab 3: Full Pipeline (M1 → M2 → M3 LLM) ─────────────────────────────

  const _BACKEND_HINTS = {
    mock:        'Mock backend requires no setup — ideal for demos and testing.',
    ollama:      'Runs locally via Ollama. Start Ollama first: ollama serve. Default model: mistral.',
    openai:      'Requires an OpenAI API key (sk-...). Default model: gpt-3.5-turbo.',
    huggingface: 'Requires a HuggingFace API key (hf_...) or uses free tier. Default model: mistralai/Mistral-7B-Instruct-v0.1.',
  };
  const _BACKEND_MODEL_DEFAULTS = {
    mock: '', ollama: 'mistral', openai: 'gpt-3.5-turbo',
    huggingface: 'mistralai/Mistral-7B-Instruct-v0.1',
  };

  function onBackendChange() {
    const b    = document.getElementById('llm-backend').value;
    const hint = document.getElementById('llm-backend-hint');
    const mdl  = document.getElementById('llm-model');
    hint.textContent = _BACKEND_HINTS[b] || '';
    mdl.placeholder  = _BACKEND_MODEL_DEFAULTS[b] || 'auto';
  }

  function useLlmExample(q) {
    document.getElementById('llm-query').value = q;
  }

  function clearLLM() {
    document.getElementById('llm-query').value = '';
    ['llm-m1-section','llm-m2-section','llm-m3-section',
     'llm-error-section','llm-pipeline-progress'].forEach(id => {
      document.getElementById(id).style.display = 'none';
    });
  }

  // Poll store status for LLM tab on first load
  (async function initLLMStoreStatus() {
    try {
      const res  = await fetch('/api/rag/status', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: '{}' });
      const data = await res.json();
      const dot    = document.getElementById('llm-store-dot');
      const status = document.getElementById('llm-store-status');
      if (data.rag_ready && data.total_chunks > 0) {
        dot.classList.add('loaded');
        status.textContent = `Vector store ready · ${data.total_chunks} chunks pre-loaded. LLM module active.`;
      } else {
        status.textContent = 'RAG module unavailable — install dependencies.';
      }
    } catch(e) { /* ignore */ }
  })();

  function _setPipelineTag(id, cls, text) {
    const el = document.getElementById(id);
    el.className = 'tag ' + cls;
    el.textContent = text;
  }

  async function runLLMPipeline() {
    const q = document.getElementById('llm-query').value.trim();
    if (!q) { document.getElementById('llm-query').focus(); return; }

    const backend = document.getElementById('llm-backend').value;
    const model   = document.getElementById('llm-model').value.trim() || null;
    const apiKey  = document.getElementById('llm-apikey').value.trim() || null;

    // Reset UI
    ['llm-m1-section','llm-m2-section','llm-m3-section','llm-error-section']
      .forEach(id => document.getElementById(id).style.display = 'none');

    // Show pipeline progress
    document.getElementById('llm-pipeline-progress').style.display = 'block';
    _setPipelineTag('prog-m1', 'warn', 'M1: Parsing…');
    _setPipelineTag('prog-m2', '',     'M2: Retrieving');
    _setPipelineTag('prog-m3', '',     'M3: Generating');

    document.getElementById('llmBtn').disabled = true;
    document.getElementById('llm-spinner').style.display = 'inline-block';

    try {
      // Animate progress tags
      await new Promise(r => setTimeout(r, 300));
      _setPipelineTag('prog-m1', 'ok', 'M1: Done ✓');
      _setPipelineTag('prog-m2', 'warn', 'M2: Retrieving…');
      await new Promise(r => setTimeout(r, 300));
      _setPipelineTag('prog-m2', 'ok', 'M2: Done ✓');
      _setPipelineTag('prog-m3', 'warn', 'M3: Generating…');

      const res  = await fetch('/api/llm/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, k: 5, backend, model, api_key: apiKey }),
      });
      const data = await res.json();

      _setPipelineTag('prog-m3', data.error ? 'error' : 'ok',
                      data.error ? 'M3: Error' : 'M3: Done ✓');

      if (data.error) showLLMError(data.error);
      else            showLLMResult(data);
    } catch (err) {
      _setPipelineTag('prog-m3', 'error', 'M3: Error');
      showLLMError('Network error: ' + err.message);
    } finally {
      document.getElementById('llmBtn').disabled = false;
      document.getElementById('llm-spinner').style.display = 'none';
    }
  }

  function showLLMError(msg) {
    document.getElementById('llm-error-msg').textContent = msg;
    document.getElementById('llm-error-section').style.display = 'block';
  }

  function showLLMResult(data) {
    const sq  = data.structured_query || {};
    const f   = sq.filters || {};
    const vt  = f.valid_time || {};
    const llm = data.llm_response || {};
    const chunks = data.chunks || [];

    // ── Module 1 summary
    const m1Items = [
      { label: 'Semantic Query',   value: sq.semantic_query || '—',    cls: '' },
      { label: 'Act',              value: f.act_name || '—',           cls: f.act_name ? 'green' : 'grey' },
      { label: 'Section',         value: f.section_id || '—',         cls: f.section_id ? '' : 'grey' },
      { label: 'Temporal Op',     value: vt.operator || '—',          cls: '' },
      { label: 'Reference Date',  value: vt.reference_date || '—',    cls: vt.reference_date ? 'highlight' : 'grey' },
      { label: 'Canonical ID',    value: sq.canonical_entity_id || '—', cls: sq.canonical_entity_id ? 'highlight' : 'grey' },
    ];
    document.getElementById('llm-m1-summary').innerHTML = m1Items.map(it => `
      <div class="summary-item">
        <div class="s-label">${it.label}</div>
        <div class="s-value ${it.cls}">${it.value}</div>
      </div>`).join('');
    document.getElementById('llm-m1-section').style.display = 'block';

    // ── Module 2 chunks
    const m2Tags = document.getElementById('llm-m2-tags');
    const dateLabel = vt.reference_date
      ? `Temporal Filter · ${vt.reference_date} ✓`
      : 'Temporal Filter · all time ✓';
    m2Tags.innerHTML = `
      <span class="tag" style="color:var(--accent2);border-color:rgba(63,185,80,.3);background:rgba(63,185,80,.1);">${dateLabel}</span>
      <span class="tag" style="color:var(--accent2);border-color:rgba(63,185,80,.3);background:rgba(63,185,80,.1);">Vector Search ✓</span>
      <span class="tag ok">${chunks.length} chunk${chunks.length===1?'':'s'}</span>`;
    document.getElementById('llm-chunk-list').innerHTML = chunks.length === 0
      ? '<div style="color:var(--muted);font-size:13px;padding:8px 0;">No chunks matched for the given time period.</div>'
      : chunks.map((c, i) => `
        <div class="chunk-card">
          <div class="chunk-meta">
            <div class="chunk-rank">${i+1}</div>
            <span class="tag" style="color:var(--accent2);background:rgba(63,185,80,.1);border-color:rgba(63,185,80,.3);">${c.regulation}</span>
            <span class="tag ok">v${c.version}</span>
            <span class="tag">${c.effective_from}${c.effective_to ? ' → '+c.effective_to : ' → now'}</span>
          </div>
          <div class="chunk-text">${escHtml(c.text)}</div>
        </div>`).join('');
    document.getElementById('llm-m2-section').style.display = 'block';

    // ── Module 3 LLM answer
    const confColor = { high: 'var(--accent2)', medium: 'var(--warn)', low: '#f85149' };
    const confBg    = { high: 'rgba(63,185,80,.1)', medium: 'rgba(210,153,34,.1)', low: 'rgba(248,81,73,.1)' };
    const conf      = (llm.confidence || 'low').toLowerCase();

    document.getElementById('llm-m3-tags').innerHTML = `
      <span class="tag" style="color:var(--purple);background:rgba(188,140,255,.1);border-color:rgba(188,140,255,.3);">Model: ${escHtml(llm.model_used || '?')}</span>
      <span class="tag" style="color:${confColor[conf]||confColor.low};background:${confBg[conf]||confBg.low};">Confidence: ${conf}</span>
      <span class="tag" style="color:var(--muted);">Prompt: ~${llm.prompt_tokens || 0} tokens</span>`;

    document.getElementById('llm-answer-text').textContent = llm.answer || '(no answer generated)';

    // Citations
    const cits = llm.cited_sections || [];
    if (cits.length > 0) {
      document.getElementById('llm-citations').innerHTML = cits.map(s =>
        `<span class="tag" style="color:var(--purple);background:rgba(188,140,255,.1);border-color:rgba(188,140,255,.3);font-size:12px;">${escHtml(s)}</span>`
      ).join('');
      document.getElementById('llm-citations-row').style.display = 'block';
    }

    // Explanation
    if (llm.explanation) {
      document.getElementById('llm-explanation').textContent = llm.explanation;
      document.getElementById('llm-explanation-row').style.display = 'block';
    }

    document.getElementById('llm-m3-section').style.display = 'block';
  }

  function colorJson(str) {
    return str
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"(\\\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\\\"])*"(\\s*:)?/g, m => {
        if (/:$/.test(m)) return `<span class="json-key">${m}</span>`;
        return `<span class="json-str">${m}</span>`;
      })
      .replace(/\\b(true|false)\\b/g, '<span class="json-bool">$1</span>')
      .replace(/\\bnull\\b/g,         '<span class="json-null">null</span>')
      .replace(/([{}\\[\\],])/g,      '<span class="json-punct">$1</span>')
      .replace(/\\b(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\b/g, '<span class="json-num">$1</span>');
  }


  // ── Tab 4: Regulation Timeline ────────────────────────────────────────────

  function useTlExample(q) { document.getElementById('tl-query').value = q; }

  function clearTimeline() {
    document.getElementById('tl-query').value = '';
    ['tl-answer-section','tl-timeline-section','tl-error-section'].forEach(id =>
      document.getElementById(id).style.display = 'none');
  }

  async function runTimeline() {
    const q = document.getElementById('tl-query').value.trim();
    if (!q) { document.getElementById('tl-query').focus(); return; }
    ['tl-answer-section','tl-timeline-section','tl-error-section'].forEach(id =>
      document.getElementById(id).style.display = 'none');
    document.getElementById('tlBtn').disabled = true;
    document.getElementById('tl-spinner').style.display = 'inline-block';
    try {
      const res  = await fetch('/api/output/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, k: 7, backend: 'mock' }),
      });
      const data = await res.json();
      if (data.error) {
        document.getElementById('tl-error-msg').textContent = data.error;
        document.getElementById('tl-error-section').style.display = 'block';
      } else showTimeline(data);
    } catch (err) {
      document.getElementById('tl-error-msg').textContent = 'Network error: ' + err.message;
      document.getElementById('tl-error-section').style.display = 'block';
    } finally {
      document.getElementById('tlBtn').disabled = false;
      document.getElementById('tl-spinner').style.display = 'none';
    }
  }

  function showTimeline(data) {
    const final  = data.final_response || {};
    const chunks = data.chunks || [];
    const basis  = final.legal_basis || [];
    const conf   = (final.confidence || 'low').toLowerCase();
    const confColor = { high:'var(--accent2)', medium:'var(--warn)', low:'#f85149' };

    // Answer card
    document.getElementById('tl-conf-tags').innerHTML = `
      <span class="tag" style="color:${confColor[conf]||confColor.low};">Confidence: ${conf}</span>
      <span class="tag" style="color:var(--warn);">Module 4 ✓</span>
      <span class="tag" style="color:var(--muted);">${(final.metadata||{}).retrieved_chunks||0} chunks</span>`;
    document.getElementById('tl-answer-text').textContent = final.answer || '(no answer)';
    document.getElementById('tl-answer-section').style.display = 'block';

    // Build timeline: group retrieved chunks by regulation, sort by effective_from
    const regGroups = {};
    chunks.forEach(c => {
      if (!regGroups[c.regulation]) regGroups[c.regulation] = [];
      regGroups[c.regulation].push(c);
    });
    // Ensure regulations from legal_basis are included even if no matching chunk
    basis.forEach(ev => {
      if (ev.regulation !== 'UNKNOWN' && !regGroups[ev.regulation])
        regGroups[ev.regulation] = [];
    });

    const regs = Object.keys(regGroups).sort();
    const tlContainer = document.getElementById('tl-timeline-container');

    if (regs.length === 0) {
      tlContainer.innerHTML = '<div style="color:var(--muted);font-size:13px;padding:8px 0;">No regulation versions found.</div>';
    } else {
      let html = '';
      regs.forEach(reg => {
        // Deduplicate chunks by version
        const seen = new Set(), deduped = [];
        (regGroups[reg] || []).forEach(c => {
          if (!seen.has(c.version)) { seen.add(c.version); deduped.push(c); }
        });
        deduped.sort((a, b) => (a.effective_from||'').localeCompare(b.effective_from||''));
        const citedVersions = new Set(basis.filter(e => e.regulation===reg).map(e => e.version));

        html += `<div style="margin-bottom:28px;">`;
        html += `<div style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
          color:var(--muted);margin-bottom:14px;display:flex;align-items:center;gap:8px;">`;
        html += `<span style="color:var(--accent);">${escHtml(reg)}</span>`;
        html += `<span class="tag" style="font-size:10px;">${deduped.length} version${deduped.length===1?'':'s'}</span>`;
        html += `</div><div class="timeline-wrapper">`;

        deduped.forEach(c => {
          const isCurrent = !c.effective_to;
          const isCited   = citedVersions.has(c.version);
          const dotCls    = isCurrent ? 'current' : 'old';
          const cardCls   = isCurrent ? 'current' : 'old';
          const dateRange = c.effective_from + (c.effective_to ? ' → ' + c.effective_to : ' → present');
          const ev = basis.find(e => e.regulation===reg && e.version===c.version);
          const excerpt = ev && ev.excerpt ? ev.excerpt : c.text.substring(0,240) + (c.text.length>240?'…':'');

          html += `<div class="tl-item"><div class="tl-dot ${dotCls}"></div>`;
          html += `<div class="tl-card ${cardCls}">`;
          html += `<div class="tl-header">`;
          html += `<span class="tl-version">v${escHtml(c.version)}</span>`;
          html += `<span class="tl-date">${escHtml(dateRange)}</span>`;
          if (isCurrent) html += `<span class="tag ok" style="font-size:10px;">Current</span>`;
          if (isCited)   html += `<span class="tag purple" style="font-size:10px;">✦ Cited</span>`;
          if (ev && ev.section) html += `<span class="tag" style="font-size:10px;color:var(--warn);background:rgba(210,153,34,.1);border-color:rgba(210,153,34,.3);">${escHtml(ev.section)}</span>`;
          html += `</div><div class="tl-excerpt">${escHtml(excerpt)}</div>`;
          html += `</div></div>`;
        });
        html += `</div></div>`;
      });
      tlContainer.innerHTML = html;
    }

    document.getElementById('tl-timeline-tags').innerHTML = `
      <span class="tag" style="color:var(--warn);">Module 4 Output ✓</span>
      <span class="tag ok">${regs.length} regulation${regs.length===1?'':'s'}</span>
      <span class="tag purple">${basis.length} citation${basis.length===1?'':'s'} resolved</span>`;
    document.getElementById('tl-timeline-section').style.display = 'block';
  }


  // ── Tab 5: Full Pipeline Flow (M1→M2→M3→M4) ──────────────────────────

  const _FP_MDL_DEFAULTS = { mock:'', ollama:'mistral', openai:'gpt-3.5-turbo',
    huggingface:'mistralai/Mistral-7B-Instruct-v0.1' };

  function onFpBackendChange() {
    const b = document.getElementById('fp-backend').value;
    document.getElementById('fp-model').placeholder = _FP_MDL_DEFAULTS[b] || 'auto';
  }
  function useFpExample(q) { document.getElementById('fp-query').value = q; }

  function clearFullPipeline() {
    document.getElementById('fp-query').value = '';
    ['fp-m1-section','fp-m2-section','fp-m3-section','fp-m4-section',
     'fp-error-section','fp-pipeline-progress'].forEach(id =>
      document.getElementById(id).style.display = 'none');
    _fpJson = '';
  }

  let _fpJson = '';

  function _setFpTag(id, cls, txt) {
    const el = document.getElementById(id);
    el.className = 'tag ' + cls; el.textContent = txt;
  }

  async function runFullPipeline() {
    const q = document.getElementById('fp-query').value.trim();
    if (!q) { document.getElementById('fp-query').focus(); return; }
    const backend = document.getElementById('fp-backend').value;
    const model   = document.getElementById('fp-model').value.trim() || null;
    const apiKey  = document.getElementById('fp-apikey').value.trim() || null;

    ['fp-m1-section','fp-m2-section','fp-m3-section','fp-m4-section','fp-error-section']
      .forEach(id => document.getElementById(id).style.display = 'none');
    _fpJson = '';

    document.getElementById('fp-pipeline-progress').style.display = 'block';
    _setFpTag('fp-prog-m1','warn','M1: Parsing…');
    _setFpTag('fp-prog-m2','','M2: Retrieving');
    _setFpTag('fp-prog-m3','','M3: Generating');
    _setFpTag('fp-prog-m4','','M4: Building');
    document.getElementById('fpBtn').disabled = true;
    document.getElementById('fp-spinner').style.display = 'inline-block';

    try {
      await new Promise(r => setTimeout(r, 280));
      _setFpTag('fp-prog-m1','ok','M1: Done ✓');
      _setFpTag('fp-prog-m2','warn','M2: Retrieving…');
      await new Promise(r => setTimeout(r, 280));
      _setFpTag('fp-prog-m2','ok','M2: Done ✓');
      _setFpTag('fp-prog-m3','warn','M3: Generating…');

      const res  = await fetch('/api/output/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, k: 5, backend, model, api_key: apiKey }),
      });
      const data = await res.json();

      if (data.error) {
        _setFpTag('fp-prog-m3','error','M3: Error');
        _setFpTag('fp-prog-m4','error','M4: Error');
        document.getElementById('fp-error-msg').textContent = data.error;
        document.getElementById('fp-error-section').style.display = 'block';
      } else {
        _setFpTag('fp-prog-m3','ok','M3: Done ✓');
        await new Promise(r => setTimeout(r, 200));
        _setFpTag('fp-prog-m4','warn','M4: Building…');
        await new Promise(r => setTimeout(r, 200));
        _setFpTag('fp-prog-m4','ok','M4: Done ✓');
        showFullPipelineResult(data);
      }
    } catch(err) {
      _setFpTag('fp-prog-m3','error','M3: Error');
      _setFpTag('fp-prog-m4','error','M4: Error');
      document.getElementById('fp-error-msg').textContent = 'Network error: '+err.message;
      document.getElementById('fp-error-section').style.display = 'block';
    } finally {
      document.getElementById('fpBtn').disabled = false;
      document.getElementById('fp-spinner').style.display = 'none';
    }
  }

  function showFullPipelineResult(data) {
    const sq    = data.structured_query || {};
    const f     = sq.filters || {};
    const vt    = f.valid_time || {};
    const llm   = data.llm_response || {};
    const final = data.final_response || {};
    const meta  = final.metadata || {};
    const chunks = data.chunks || [];
    const basis  = final.legal_basis || [];
    const conf   = (final.confidence || 'low').toLowerCase();
    const confColor = { high:'var(--accent2)', medium:'var(--warn)', low:'#f85149' };

    // ── Module 1
    document.getElementById('fp-m1-grid').innerHTML = [
      { label:'Semantic Query',  value: sq.semantic_query||'—', cls:'' },
      { label:'Act',             value: f.act_name||'—', cls: f.act_name?'green':'grey' },
      { label:'Section',         value: f.section_id||'—', cls: f.section_id?'':'grey' },
      { label:'Temporal Op',     value: vt.operator||'—', cls:'' },
      { label:'Reference Date',  value: vt.reference_date||'—', cls: vt.reference_date?'highlight':'grey' },
      { label:'Canonical ID',    value: sq.canonical_entity_id||'—', cls: sq.canonical_entity_id?'highlight':'grey' },
    ].map(it => `<div class="summary-item"><div class="s-label">${it.label}</div><div class="s-value ${it.cls}">${it.value}</div></div>`).join('');
    document.getElementById('fp-m1-section').style.display = 'block';

    // ── Module 2
    document.getElementById('fp-m2-tags').innerHTML = `
      <span class="tag" style="color:var(--accent2);border-color:rgba(63,185,80,.3);background:rgba(63,185,80,.1);">${vt.reference_date?'Filter · '+vt.reference_date+' ✓':'All time ✓'}</span>
      <span class="tag ok">${chunks.length} chunk${chunks.length===1?'':'s'}</span>`;
    document.getElementById('fp-chunk-list').innerHTML = chunks.length===0
      ? '<div style="color:var(--muted);font-size:13px;padding:8px 0;">No chunks matched.</div>'
      : chunks.map((c,i) => `<div class="chunk-card">
          <div class="chunk-meta"><div class="chunk-rank">${i+1}</div>
            <span class="tag" style="color:var(--accent2);background:rgba(63,185,80,.1);border-color:rgba(63,185,80,.3);">${c.regulation}</span>
            <span class="tag ok">v${c.version}</span>
            <span class="tag">${c.effective_from}${c.effective_to?' → '+c.effective_to:' → now'}</span></div>
          <div class="chunk-text">${escHtml(c.text.substring(0,200))}${c.text.length>200?'…':''}</div></div>`).join('');
    document.getElementById('fp-m2-section').style.display = 'block';

    // ── Module 3
    document.getElementById('fp-m3-tags').innerHTML = `
      <span class="tag" style="color:var(--purple);background:rgba(188,140,255,.1);border-color:rgba(188,140,255,.3);">Model: ${escHtml(llm.model_used||'?')}</span>
      <span class="tag" style="color:${confColor[conf]||confColor.low};">Confidence: ${conf}</span>`;
    document.getElementById('fp-answer-text').textContent = llm.answer||'(no answer)';
    const cits = llm.cited_sections||[];
    if (cits.length>0) {
      document.getElementById('fp-citations').innerHTML = cits.map(s =>
        `<span class="tag" style="color:var(--purple);background:rgba(188,140,255,.1);border-color:rgba(188,140,255,.3);font-size:12px;">${escHtml(s)}</span>`).join('');
      document.getElementById('fp-citations-row').style.display = 'block';
    }
    document.getElementById('fp-m3-section').style.display = 'block';

    // ── Module 4
    document.getElementById('fp-m4-tags').innerHTML = `
      <span class="tag" style="color:var(--warn);background:rgba(210,153,34,.1);border-color:rgba(210,153,34,.3);">Output Module ✓</span>
      <span class="tag" style="color:${confColor[conf]||confColor.low};">Confidence: ${conf}</span>
      <span class="tag" style="color:var(--muted);">${meta.retrieved_chunks||0} chunks · ${basis.length} evidence</span>`;

    // Legal Basis evidence cards
    const lbEl = document.getElementById('fp-legal-basis');
    lbEl.innerHTML = basis.length===0
      ? '<div style="color:var(--muted);font-size:13px;padding:8px 0;">No citations resolved.</div>'
      : basis.map(ev => {
          const dateRange = ev.effective_from+(ev.effective_to?' → '+ev.effective_to:' → present');
          return `<div class="evidence-card"><div class="evidence-meta">
            <span class="tag" style="color:var(--warn);background:rgba(210,153,34,.1);border-color:rgba(210,153,34,.3);">${escHtml(ev.regulation)}</span>
            <span class="tag ok">v${escHtml(ev.version)}</span>
            <span class="tag" style="font-size:11px;">${escHtml(ev.section)}</span>
            <span class="tag" style="font-size:10px;color:var(--muted);">${escHtml(dateRange)}</span></div>
            ${ev.excerpt?`<div class="evidence-excerpt">${escHtml(ev.excerpt)}</div>`:''}</div>`;
        }).join('');

    if (final.explanation) {
      document.getElementById('fp-explanation').textContent = final.explanation;
      document.getElementById('fp-explanation-row').style.display = 'block';
    }

    _fpJson = JSON.stringify(final, null, 2);
    document.getElementById('fp-json-out').innerHTML = colorJson(_fpJson);
    document.getElementById('fp-m4-section').style.display = 'block';
  }

  function copyFpJson() {
    if (!_fpJson) return;
    navigator.clipboard.writeText(_fpJson).then(() => {
      const b = document.getElementById('fp-copy-btn');
      b.textContent = 'Copied!';
      setTimeout(() => b.textContent = 'Copy JSON', 1500);
    });
  }

</script>

</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):  # silence default access log
        pass

    def _send(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, "text/html; charset=utf-8", _HTML.encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        # ── /api/parse (Module 1) ────────────────────────────────────────────
        if self.path == "/api/parse":
            try:
                payload = json.loads(body)
                raw_q   = payload.get("query", "").strip()
                if not raw_q:
                    raise ValueError("Query is empty.")
                sq   = process_query(raw_q)
                data = sq.to_dict()
                resp = json.dumps(data, indent=2).encode()
                self._send(200, "application/json; charset=utf-8", resp)
            except Exception as exc:
                err = json.dumps({"error": str(exc)}).encode()
                self._send(200, "application/json; charset=utf-8", err)

        # ── /api/rag/ingest (Module 2 – step 1) ─────────────────────────────
        elif self.path == "/api/rag/ingest":
            global _rag_store, _rag_chunks, _rag_retriever
            try:
                if not _rag_ready:
                    raise RuntimeError("RAG module not available. Install: pip install faiss-cpu sentence-transformers")
                from rag_module import (
                    ingest_document,
                    chunk_document,
                    embed_chunks,
                    VectorStore,
                    TemporalRAGRetriever,
                )

                payload = json.loads(body)
                doc_data = payload.get("document", {})
                if not doc_data:
                    raise ValueError("No document provided.")

                doc    = ingest_document(doc_data)
                chunks = chunk_document(doc)
                embed_chunks(chunks)

                # Rebuild store with accumulated chunks
                _rag_chunks.extend(chunks)
                _rag_store = VectorStore()
                _rag_store.add_chunks(_rag_chunks)
                _rag_retriever = TemporalRAGRetriever(_rag_store, _rag_chunks)

                resp = json.dumps({
                    "status":       "ok",
                    "regulation":   doc.regulation,
                    "version":      doc.version,
                    "chunks_added": len(chunks),
                    "total_chunks": len(_rag_chunks),
                }).encode()
                self._send(200, "application/json; charset=utf-8", resp)

            except Exception as exc:
                err = json.dumps({"error": str(exc)}).encode()
                self._send(200, "application/json; charset=utf-8", err)

        # ── /api/rag/status ─────────────────────────────────────────────────
        elif self.path == "/api/rag/status":
            resp = json.dumps({
                "rag_ready":    _rag_ready,
                "total_chunks": len(_rag_chunks),
            }).encode()
            self._send(200, "application/json; charset=utf-8", resp)

        # ── /api/rag/pipeline (Full end-to-end: Module 1 → Module 2) ─────────
        elif self.path == "/api/rag/pipeline":
            try:
                if not _rag_ready or _rag_retriever is None:
                    raise RuntimeError(
                        "RAG module not initialised. "
                        "Install dependencies: pip install faiss-cpu sentence-transformers"
                    )

                payload = json.loads(body)
                raw_q   = payload.get("query", "").strip()
                k       = int(payload.get("k", 5))

                if not raw_q:
                    raise ValueError("Query is empty.")

                # Module 1: parse natural-language query → StructuredQuery
                sq = process_query(raw_q)

                # Module 2: retrieve using the StructuredQuery
                # retrieve_from_structured extracts semantic_query and
                # reference_date automatically from the StructuredQuery.
                results = _rag_retriever.retrieve_from_structured(sq, k=k)

                resp = json.dumps({
                    "raw_query":       raw_q,
                    "structured_query": sq.to_dict(),
                    "chunks": [
                        {
                            "chunk_id":       c.chunk_id,
                            "regulation":     c.regulation,
                            "version":        c.version,
                            "effective_from": c.effective_from,
                            "effective_to":   c.effective_to,
                            "text":           c.text,
                        }
                        for c in results
                    ],
                }, indent=2).encode()
                self._send(200, "application/json; charset=utf-8", resp)

            except Exception as exc:
                err = json.dumps({"error": str(exc)}).encode()
                self._send(200, "application/json; charset=utf-8", err)


        # ── /api/rag/retrieve (legacy: raw query + manual date) ───────────────
        elif self.path == "/api/rag/retrieve":
            try:
                if not _rag_ready or _rag_retriever is None:
                    raise RuntimeError("RAG module not available.")
                if not _rag_chunks:
                    raise ValueError("Vector store is empty.")

                payload    = json.loads(body)
                query      = payload.get("query", "").strip()
                query_date = payload.get("query_date") or None
                k          = int(payload.get("k", 5))

                if not query:
                    raise ValueError("Query is empty.")

                results = _rag_retriever.retrieve(query, query_date=query_date, k=k)

                resp = json.dumps({
                    "query":      query,
                    "query_date": query_date,
                    "chunks": [
                        {
                            "chunk_id":       c.chunk_id,
                            "regulation":     c.regulation,
                            "version":        c.version,
                            "effective_from": c.effective_from,
                            "effective_to":   c.effective_to,
                            "text":           c.text,
                        }
                        for c in results
                    ],
                }, indent=2).encode()
                self._send(200, "application/json; charset=utf-8", resp)

            except Exception as exc:
                err = json.dumps({"error": str(exc)}).encode()
                self._send(200, "application/json; charset=utf-8", err)

        # ── /api/llm/generate (Full pipeline: Module 1 → Module 2 → Module 3) ─
        elif self.path == "/api/llm/generate":
            try:
                if not _rag_ready or _rag_retriever is None:
                    raise RuntimeError(
                        "RAG module not initialised. "
                        "Install dependencies: pip install faiss-cpu sentence-transformers"
                    )

                payload  = json.loads(body)
                raw_q    = payload.get("query", "").strip()
                k        = int(payload.get("k", 5))
                backend  = payload.get("backend", "mock") or "mock"
                model    = payload.get("model") or None
                api_key  = payload.get("api_key") or None

                if not raw_q:
                    raise ValueError("Query is empty.")

                # Module 1: natural-language query → StructuredQuery
                sq = process_query(raw_q)

                # Module 2: retrieve temporally-filtered regulation chunks
                results = _rag_retriever.retrieve_from_structured(sq, k=k)

                # Module 3: generate a grounded, cited LLM answer
                from llm_module import generate_answer
                llm_response = generate_answer(
                    sq,
                    results,
                    backend=backend,
                    model=model,
                    api_key=api_key,
                )

                resp = json.dumps({
                    "raw_query":        raw_q,
                    "structured_query": sq.to_dict(),
                    "chunks": [
                        {
                            "chunk_id":       c.chunk_id,
                            "regulation":     c.regulation,
                            "version":        c.version,
                            "effective_from": c.effective_from,
                            "effective_to":   c.effective_to,
                            "text":           c.text,
                        }
                        for c in results
                    ],
                    "llm_response": llm_response.to_dict(),
                }, indent=2).encode()
                self._send(200, "application/json; charset=utf-8", resp)

            except Exception as exc:
                err = json.dumps({"error": str(exc)}).encode()
                self._send(200, "application/json; charset=utf-8", err)

        # ── /api/output/generate (Full pipeline: M1 → M2 → M3 → M4) ────────────
        elif self.path == "/api/output/generate":
            try:
                if not _rag_ready or _rag_retriever is None:
                    raise RuntimeError(
                        "RAG module not initialised. "
                        "Install dependencies: pip install faiss-cpu sentence-transformers"
                    )

                payload  = json.loads(body)
                raw_q    = payload.get("query", "").strip()
                k        = int(payload.get("k", 5))
                backend  = payload.get("backend", "mock") or "mock"
                model    = payload.get("model") or None
                api_key  = payload.get("api_key") or None

                if not raw_q:
                    raise ValueError("Query is empty.")

                # Module 1: natural-language query → StructuredQuery
                sq = process_query(raw_q)

                # Module 2: retrieve temporally-filtered regulation chunks
                results = _rag_retriever.retrieve_from_structured(sq, k=k)

                # Module 3: generate a grounded, cited LLM answer
                from llm_module import generate_answer
                llm_response = generate_answer(
                    sq, results,
                    backend=backend, model=model, api_key=api_key,
                )

                # Module 4: build traceable FinalSystemResponse
                from output_module import build_response
                final = build_response(llm_response, results)

                resp = json.dumps({
                    "raw_query":        raw_q,
                    "structured_query": sq.to_dict(),
                    "llm_response":     llm_response.to_dict(),
                    "final_response":   final.to_dict(),
                    "chunks": [
                        {
                            "chunk_id":       c.chunk_id,
                            "regulation":     c.regulation,
                            "version":        c.version,
                            "effective_from": c.effective_from,
                            "effective_to":   c.effective_to,
                            "text":           c.text,
                        }
                        for c in results
                    ],
                }, indent=2).encode()
                self._send(200, "application/json; charset=utf-8", resp)

            except Exception as exc:
                err = json.dumps({"error": str(exc)}).encode()
                self._send(200, "application/json; charset=utf-8", err)

        else:
            self._send(404, "application/json", b'{"error":"not found"}')



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PORT = 8765
    server = HTTPServer(("localhost", PORT), _Handler)
    url = f"http://localhost:{PORT}"
    print(f"  TDDR Demo UI  ->  {url}")
    print("  Press Ctrl+C to stop.\n")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
