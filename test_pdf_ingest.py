"""Quick diagnostic: can the RAG pipeline ingest the MVA PDFs?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rag_module import ingest_document, chunk_document, embed_chunks

data_dir = os.path.join(os.path.dirname(__file__), "data")
print(f"Data dir: {data_dir}")
print(f"Exists: {os.path.isdir(data_dir)}")

import glob
pdfs = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
print(f"PDFs found: {len(pdfs)}")
for p in pdfs:
    print(f"  - {os.path.basename(p)}")

print()
for pdf_path in pdfs:
    name = os.path.basename(pdf_path)
    try:
        doc = ingest_document(pdf_path)
        print(f"✓ {name}")
        print(f"    regulation={doc.regulation}, version={doc.version}")
        print(f"    effective_from={doc.effective_from}, effective_to={doc.effective_to}")
        print(f"    text length: {len(doc.text)} chars")
        print(f"    first 200 chars: {doc.text[:200]!r}")
        
        chunks = chunk_document(doc)
        print(f"    chunks: {len(chunks)}")
        if chunks:
            print(f"    chunk[0].regulation = {chunks[0].regulation!r}")
            print(f"    chunk[0].text[:100] = {chunks[0].text[:100]!r}")
    except Exception as e:
        print(f"✗ {name}: {e}")
    print()
