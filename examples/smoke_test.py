"""快速烟雾测试：真实 BGE OpenVINO + FAISS 端到端检索流程"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from doc_parser import main as parse
from vector_store import main as build
from rag_query import main as query

SAMPLE = os.path.join(os.path.dirname(__file__), "sample.txt")

r = json.loads(parse({"file_path": SAMPLE, "chunk_size": 120}))
print("PARSE:", r["status"], "chunks =", r.get("total_chunks"))
doc_id = r["doc_id"]

r = json.loads(build({"doc_id": doc_id}))
print("BUILD:", r["status"], "vectors =", r.get("total_vectors"), "dim =", r.get("vector_dim"))

r = json.loads(query({"doc_id": doc_id, "query": "OpenVINO 支持哪些硬件设备", "top_k": 3}))
print("QUERY:", r["status"])
for hit in r.get("results", []):
    print(f"  rank={hit['rank']} score={hit['score']:.4f} text={hit['text'][:80]}...")
