"""
工具3: RAG 查询 (rag_query.py)
功能：对已构建索引的文档进行语义检索，返回最相关的文本片段
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding import load_embedding_model, embed_text


def main(args):
    """
    入口函数。被 Agent 框架调用。
    args: {"doc_id": "...", "query": "...", "top_k": 3, "device": "CPU"}
    """
    doc_id = args.get("doc_id", "")
    query = args.get("query", "")
    top_k = args.get("top_k", 3)
    device = args.get("device", "CPU")

    if not doc_id:
        return json.dumps(
            {"status": "error", "message": "缺少 doc_id 参数"}, ensure_ascii=False
        )
    if not query:
        return json.dumps(
            {"status": "error", "message": "缺少 query 参数"}, ensure_ascii=False
        )

    # 加载分块数据
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data"
    )
    chunks_file = os.path.join(data_dir, f"{doc_id}_chunks.json")

    if not os.path.exists(chunks_file):
        return json.dumps(
            {"status": "error", "message": f"文档未解析: {doc_id}"},
            ensure_ascii=False,
        )

    with open(chunks_file, "r", encoding="utf-8") as f:
        doc_data = json.load(f)
    chunks = doc_data.get("chunks", [])

    # 加载 FAISS 索引
    import faiss

    index_file = os.path.join(data_dir, f"{doc_id}.index")
    if not os.path.exists(index_file):
        return json.dumps(
            {
                "status": "error",
                "message": f"向量索引不存在，请先调用 build_index: {doc_id}",
            },
            ensure_ascii=False,
        )
    index = faiss.read_index(index_file)

    # 真实查询向量
    model, error = load_embedding_model(device=device)
    if error:
        return json.dumps(
            {"status": "error", "message": error}, ensure_ascii=False
        )

    try:
        query_emb = embed_text(model, query)
    except Exception as e:
        return json.dumps(
            {"status": "error", "message": f"查询编码失败: {e}"},
            ensure_ascii=False,
        )

    query_emb = query_emb.reshape(1, -1).astype(np.float32)

    actual_k = min(int(top_k), len(chunks))
    scores, indices = index.search(query_emb, actual_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(chunks):
            results.append(
                {
                    "rank": i + 1,
                    # IndexFlatIP 返回内积，归一化向量下即余弦相似度（越大越相关）
                    "score": float(scores[0][i]),
                    "chunk_index": int(idx),
                    "text": chunks[idx],
                }
            )

    return json.dumps(
        {
            "status": "success",
            "doc_id": doc_id,
            "query": query,
            "device": device,
            "total_results": len(results),
            "results": results,
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 查询工具")
    parser.add_argument("doc_id", help="文档ID")
    parser.add_argument("query", help="查询问题")
    parser.add_argument("--top-k", type=int, default=3, help="返回片段数量")
    parser.add_argument("--device", default="CPU", help="OpenVINO 设备 CPU/GPU/NPU")
    args = parser.parse_args()

    result = main(
        {
            "doc_id": args.doc_id,
            "query": args.query,
            "top_k": args.top_k,
            "device": args.device,
        }
    )
    print(result)
