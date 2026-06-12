"""
工具2: 向量存储 (vector_store.py)
功能：对文档分块构建 FAISS 向量索引，使用 OpenVINO 优化的 BGE embedding
"""
import os
import sys
import json
import numpy as np

# 允许作为脚本或模块运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding import load_embedding_model, embed_texts


def build_faiss_index(embeddings):
    """构建 FAISS 内积索引（embeddings 已 L2 归一化，等价于余弦相似度）"""
    try:
        import faiss
    except ImportError:
        return None, "缺少 faiss-cpu，请运行: pip install faiss-cpu"

    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        return index, None
    except Exception as e:
        return None, f"构建索引失败: {e}"


def main(args):
    """
    入口函数。被 Agent 框架调用。
    args: {"doc_id": "...", "device": "CPU"}
    """
    doc_id = args.get("doc_id", "")
    device = args.get("device", "CPU")

    if not doc_id:
        return json.dumps(
            {"status": "error", "message": "缺少 doc_id 参数"}, ensure_ascii=False
        )

    # 加载分块数据
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data"
    )
    chunks_file = os.path.join(data_dir, f"{doc_id}_chunks.json")

    if not os.path.exists(chunks_file):
        return json.dumps(
            {
                "status": "error",
                "message": f"文档未解析，请先调用 parse_document: {doc_id}",
            },
            ensure_ascii=False,
        )

    with open(chunks_file, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    chunks = doc_data.get("chunks", [])
    if not chunks:
        return json.dumps(
            {"status": "error", "message": "文档无文本块"}, ensure_ascii=False
        )

    # 加载 OpenVINO BGE 模型
    model, error = load_embedding_model(device=device)
    if error:
        return json.dumps(
            {"status": "error", "message": error}, ensure_ascii=False
        )

    # 真实批量编码
    try:
        embeddings = embed_texts(model, chunks)
    except Exception as e:
        return json.dumps(
            {"status": "error", "message": f"向量化失败: {e}"},
            ensure_ascii=False,
        )

    if embeddings.size == 0:
        return json.dumps(
            {"status": "error", "message": "向量化结果为空"}, ensure_ascii=False
        )

    # 构建 FAISS 索引
    index, error = build_faiss_index(embeddings)
    if error:
        return json.dumps(
            {"status": "error", "message": error}, ensure_ascii=False
        )

    # 持久化索引 + 向量
    index_file = os.path.join(data_dir, f"{doc_id}.index")
    import faiss

    faiss.write_index(index, index_file)
    emb_file = os.path.join(data_dir, f"{doc_id}_embeddings.npy")
    np.save(emb_file, embeddings)

    return json.dumps(
        {
            "status": "success",
            "doc_id": doc_id,
            "device": device,
            "total_vectors": int(embeddings.shape[0]),
            "vector_dim": int(embeddings.shape[1]),
            "index_file": index_file,
            "message": f"向量索引构建完成，共 {embeddings.shape[0]} 个向量",
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="向量索引构建工具")
    parser.add_argument("doc_id", help="文档ID")
    parser.add_argument("--device", default="CPU", help="OpenVINO 设备 CPU/GPU/NPU")
    args = parser.parse_args()

    result = main({"doc_id": args.doc_id, "device": args.device})
    print(result)
