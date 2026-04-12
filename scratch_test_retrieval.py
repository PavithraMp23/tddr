import sys
import os

# Append the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from demo_ui import _rag_retriever, _init_rag
from user_input_module.user_input_module import process_query

print("Initializing RAG (may take a few seconds)...")
# We assume demo_ui's _init_rag has already been called if we imported it, 
# but if it didn't finish, we might need to wait. Since it's run on import in demo_ui, 
# _rag_retriever should be populated.

query = "Check difference between feb and july amendment for HOWM"
print(f"\nQUERY: {query}")

try:
    sq = process_query(query)
    
    # Run retrieval
    chunks = _rag_retriever.retrieve_from_structured(sq, k=5)
    print("\n--- RETRIEVED CHUNKS ---")
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1}: version='{c.version}' | text='{c.text[:80]}...'")
        
except Exception as e:
    print(f"Error: {e}")
