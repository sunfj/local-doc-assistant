"""
工具1: 文档解析 (doc_parser.py)
功能：解析 PDF/TXT 文件，提取文本并分块
"""
import os
import sys
import json
import hashlib
import uuid

def chunk_text(text, chunk_size=500, overlap=50):
    """将文本分块，带重叠以保持上下文连贯"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks

def extract_pdf(file_path):
    """提取 PDF 文本"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        return None, "缺少 pymupdf，请运行: pip install pymupdf"
    except Exception as e:
        return None, f"PDF 解析失败: {str(e)}"

def extract_txt(file_path):
    """提取 TXT 文本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return None, f"TXT 读取失败: {str(e)}"

def main(args):
    """
    入口函数。被 Agent 框架调用。
    args: {"file_path": "...", "chunk_size": 500}
    """
    file_path = args.get("file_path", "")
    chunk_size = args.get("chunk_size", 500)
    
    if not file_path or not os.path.exists(file_path):
        return json.dumps({
            "status": "error",
            "message": f"文件不存在: {file_path}"
        }, ensure_ascii=False)
    
    # 根据扩展名选择解析器
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        text, error = extract_pdf(file_path)
        if error:
            return json.dumps({"status": "error", "message": error}, ensure_ascii=False)
    elif ext in ('.txt', '.md', '.markdown'):
        text, error = extract_txt(file_path), None
        if not text:
            return json.dumps({"status": "error", "message": "文本为空或读取失败"}, ensure_ascii=False)
    else:
        return json.dumps({
            "status": "error",
            "message": f"不支持的文件格式: {ext}，支持 .pdf, .txt, .md"
        }, ensure_ascii=False)
    
    # 生成文档 ID（基于文件路径的哈希）
    doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
    
    # 分块
    chunks = chunk_text(text, chunk_size=chunk_size)
    
    # 保存文本块到本地（供后续向量化使用）
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    chunks_file = os.path.join(data_dir, f"{doc_id}_chunks.json")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump({
            "doc_id": doc_id,
            "file_path": file_path,
            "total_chunks": len(chunks),
            "chunks": chunks
        }, f, ensure_ascii=False, indent=2)
    
    return json.dumps({
        "status": "success",
        "doc_id": doc_id,
        "file_name": os.path.basename(file_path),
        "total_chars": len(text),
        "total_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunks_file": chunks_file
    }, ensure_ascii=False)


if __name__ == "__main__":
    # 命令行测试
    import argparse
    parser = argparse.ArgumentParser(description="文档解析工具")
    parser.add_argument("file_path", help="文档路径")
    parser.add_argument("--chunk-size", type=int, default=500, help="分块大小")
    args = parser.parse_args()
    
    result = main({"file_path": args.file_path, "chunk_size": args.chunk_size})
    print(result)
