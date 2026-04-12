"""Trace the full M1 → M2 pipeline for an MVA query to find the breakdown."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from user_input_module import process_query
from rag_module import (
    VectorStore, TemporalRAGRetriever,
    ingest_document, chunk_document, embed_chunks,
)
import glob

# ---------- Build the same store as demo_ui.py ----------
print("=== Building vector store ===")
all_chunks = []
store = VectorStore()

# Dummy docs (same as demo_ui.py uses)
dummy_docs = [
    {"regulation": "IPC", "version": "1860",
     "effective_from": "1860-01-01", "effective_to": "2017-12-31",
     "text": "Section 302 Punishment for Murder\nWhoever commits murder shall be punished with death or imprisonment for life."},
    {"regulation": "IPC", "version": "2018",
     "effective_from": "2018-01-01", "effective_to": None,
     "text": "Section 302 Punishment for Murder (Unchanged)\nWhoever commits murder shall be punished with death or imprisonment for life."},
]
for d in dummy_docs:
    doc = ingest_document(d)
    chunks = chunk_document(doc)
    embed_chunks(chunks)
    all_chunks.extend(chunks)
print(f"  Dummy chunks: {len(all_chunks)}")

# PDF docs
data_dir = os.path.join(os.path.dirname(__file__), "data")
for pdf_path in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
    doc = ingest_document(pdf_path)
    chunks = chunk_document(doc)
    embed_chunks(chunks)
    all_chunks.extend(chunks)
    print(f"  {os.path.basename(pdf_path)}: {len(chunks)} chunks (reg={doc.regulation})")

store.add_chunks(all_chunks)
retriever = TemporalRAGRetriever(store, all_chunks)
print(f"  Total chunks: {len(all_chunks)}")

# Count by regulation
from collections import Counter
reg_counts = Counter(c.regulation for c in all_chunks)
print(f"  By regulation: {dict(reg_counts)}")

# ---------- Run Module 1 ----------
print("\n=== Module 1: process_query ===")
query = "What is the penalty for drunk driving under MVA?"
sq = process_query(query)
print(f"  Query: {query}")
print(f"  semantic_query: {sq.semantic_query}")
print(f"  act_name: {sq.filters.act_name}")
print(f"  canonical_entity_id: {sq.filters.canonical_entity_id}")
print(f"  canonical_entity_ids: {sq.filters.canonical_entity_ids}")
print(f"  temporal operator: {sq.filters.valid_time.operator}")
print(f"  reference_date: {sq.filters.valid_time.reference_date}")

# ---------- Run Module 2 ----------
print("\n=== Module 2: retrieve_from_structured ===")
results = retriever.retrieve_from_structured(sq, k=5)
print(f"  Results: {len(results)}")
for i, chunk in enumerate(results):
    print(f"  [{i+1}] regulation={chunk.regulation}, version={chunk.version}, "
          f"chunk_id={chunk.chunk_id}")
    print(f"       text[:120] = {chunk.text[:120]!r}")

# ---------- Also test direct retrieve with regulation_filter ----------
print("\n=== Direct retrieve with regulation_filter='MVA' ===")
results2 = retriever.retrieve("penalty for drunk driving motor vehicles act", regulation_filter="MVA", k=5)
print(f"  Results: {len(results2)}")
for i, chunk in enumerate(results2):
    print(f"  [{i+1}] regulation={chunk.regulation}, version={chunk.version}")
    print(f"       text[:120] = {chunk.text[:120]!r}")
