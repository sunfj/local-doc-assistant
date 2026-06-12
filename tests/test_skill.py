"""
测试用例：验证 Skill 的基本功能
"""
import os
import sys
import json
import tempfile

# 添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from doc_parser import main as parse_main
from rag_query import main as query_main

def test_parse_txt():
    """测试 TXT 文件解析"""
    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("OpenVINO是英特尔的开源推理框架。它支持CPU、GPU和NPU设备。")
        f.write("OpenVINO可以将模型转换为优化的IR格式。")
        f.write("OpenVINO GenAI提供了生成式AI的专用API。")
        test_file = f.name
    
    try:
        result = json.loads(parse_main({"file_path": test_file, "chunk_size": 50}))
        assert result["status"] == "success"
        assert result["total_chunks"] > 0
        assert "doc_id" in result
        print(f"✓ TXT 解析测试通过: {result['total_chunks']} chunks")
        return result["doc_id"]
    finally:
        os.unlink(test_file)

def test_parse_pdf_not_found():
    """测试不存在的文件"""
    result = json.loads(parse_main({"file_path": "/nonexistent/file.pdf"}))
    assert result["status"] == "error"
    print("✓ 文件不存在测试通过")

def test_parse_unsupported_format():
    """测试不支持的格式"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False) as f:
        f.write("test")
        test_file = f.name
    
    try:
        result = json.loads(parse_main({"file_path": test_file}))
        assert result["status"] == "error"
        print("✓ 不支持格式测试通过")
    finally:
        os.unlink(test_file)

def test_query_no_index():
    """测试未构建索引的查询"""
    result = json.loads(query_main({
        "doc_id": "nonexistent",
        "query": "测试查询"
    }))
    assert result["status"] == "error"
    print("✓ 无索引查询测试通过")

if __name__ == "__main__":
    print("Running Skill tests...\n")
    
    test_parse_txt()
    test_parse_pdf_not_found()
    test_parse_unsupported_format()
    test_query_no_index()
    
    print("\n✓ All tests passed!")
