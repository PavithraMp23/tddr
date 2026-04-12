import sys
import os

# Append the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from user_input_module.user_input_module import process_query

query = "Check difference between feb and july amendment for HOWM"
print(f"QUERY: {query}")

try:
    sq = process_query(query)
    print("\n--- STRUCTURED QUERY ---")
    print(f"Act Name: {sq.filters.act_name}")
    print(f"Canonical ID: {sq.filters.canonical_entity_id}")
    print(f"Version Refs: {sq.filters.version_refs}")
    print(f"Valid Time: {sq.filters.valid_time.operator}")
except Exception as e:
    print(f"Error parsing query: {e}")
