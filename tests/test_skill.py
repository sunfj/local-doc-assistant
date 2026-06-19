"""
Skill 测试用例
- 基础文档解析
- 错误输入处理
- 真实 BGE(OpenVINO) + FAISS 检索链路（若模型已导出）
- run_agent 工具注册与 schema 转换
"""
import os
import sys
import json
import tempfile
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS)

from doc_parser import main as parse_main
from vector_store import main as build_main
from rag_query import main as query_main
from embedding import default_model_dir
from run_agent import load_manifest, manifest_to_openai_tools, build_tool_registry


def test_parse_txt():
    """测试 TXT 文件解析"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("OpenVINO是英特尔的开源推理框架。它支持CPU、GPU和NPU设备。")
        f.write("OpenVINO可以将模型转换为优化的IR格式。")
        f.write("OpenVINO GenAI提供了生成式AI的专用API。")
        test_file = f.name

    try:
        result = json.loads(parse_main({"file_path": test_file, "chunk_size": 50}))
        assert result["status"] == "success"
        assert result["total_chunks"] > 0
        assert "doc_id" in result
    finally:
        os.unlink(test_file)


def test_parse_pdf_not_found():
    """测试不存在的文件"""
    result = json.loads(parse_main({"file_path": "/nonexistent/file.pdf"}))
    assert result["status"] == "error"
    assert "文件不存在" in result["message"]


def test_parse_unsupported_format():
    """测试不支持的格式"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".docx", delete=False) as f:
        f.write("test")
        test_file = f.name

    try:
        result = json.loads(parse_main({"file_path": test_file}))
        assert result["status"] == "error"
        assert "不支持" in result["message"]
    finally:
        os.unlink(test_file)


def test_query_no_index():
    """测试未构建索引的查询"""
    result = json.loads(query_main({"doc_id": "nonexistent", "query": "测试查询"}))
    assert result["status"] == "error"


def test_manifest_to_openai_tools():
    """manifest.json 可转换为 OpenAI/Ollama tool schema"""
    manifest = load_manifest(os.path.join(ROOT, "manifest.json"))
    tools = manifest_to_openai_tools(manifest)
    names = [t["function"]["name"] for t in tools]
    assert names == ["parse_document", "build_index", "query_document"]
    assert all(t["type"] == "function" for t in tools)
    assert tools[0]["function"]["parameters"]["type"] == "object"


def test_tool_registry():
    """entry_point 可动态加载为可调用函数"""
    manifest = load_manifest(os.path.join(ROOT, "manifest.json"))
    registry = build_tool_registry(manifest)
    assert set(registry) == {"parse_document", "build_index", "query_document"}
    assert all(callable(v) for v in registry.values())


@pytest.mark.skipif(
    not os.path.exists(os.path.join(default_model_dir(), "openvino_model.xml")),
    reason="BGE OpenVINO 模型未导出，跳过真实 RAG 链路测试",
)
def test_real_rag_retrieval_with_openvino_bge():
    """真实 BGE(OpenVINO) + FAISS 链路：应命中硬件设备段落"""
    sample = os.path.join(ROOT, "examples", "sample.txt")
    parsed = json.loads(parse_main({"file_path": sample, "chunk_size": 120}))
    assert parsed["status"] == "success"

    built = json.loads(build_main({"doc_id": parsed["doc_id"], "device": "CPU"}))
    assert built["status"] == "success"
    assert built["vector_dim"] > 0

    result = json.loads(
        query_main(
            {
                "doc_id": parsed["doc_id"],
                "query": "OpenVINO 支持哪些硬件设备？",
                "top_k": 3,
                "device": "CPU",
            }
        )
    )
    assert result["status"] == "success"
    assert result["total_results"] == 3
    joined = "\n".join(hit["text"] for hit in result["results"])
    assert "CPU" in joined and "GPU" in joined


if __name__ == "__main__":
    # 兼容直接 python tests/test_skill.py 的执行方式
    test_parse_txt()
    test_parse_pdf_not_found()
    test_parse_unsupported_format()
    test_query_no_index()
    test_manifest_to_openai_tools()
    test_tool_registry()
    if os.path.exists(os.path.join(default_model_dir(), "openvino_model.xml")):
        test_real_rag_retrieval_with_openvino_bge()
    print("All tests passed")
