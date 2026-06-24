"""
工具1: 文档解析 (doc_parser.py)
功能：解析 PDF/TXT/图片文件，提取文本并分块
- 文字版 PDF：直接用 PyMuPDF 提取
- 扫描件 PDF（无文字层）：自动路由到 PaddleOCR
- 图片（PNG/JPG/BMP/TIFF）：直接 OCR
- TXT/Markdown：直接读取
"""
import os
import sys
import json
import hashlib
import uuid

def chunk_text(text, chunk_size=500, overlap=50):
    """将文本分块，带重叠以保持上下文连贯"""
    if not text:
        return []
    # 限制重叠不超过 chunk_size 的一半，避免步长过小；保证每次至少前进 1
    overlap = max(0, min(overlap, chunk_size // 2))
    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += step
    return chunks

def extract_pdf(file_path, auto_ocr=True):
    """
    提取 PDF 文本，返回 (text_or_None, error_or_None)
    auto_ocr=True 时，若 PyMuPDF 提取到的文字极少（扫描件），自动调 OCR
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # 检测是否为扫描件：文字极少，需要 OCR
        if auto_ocr and len(text.strip()) < 50:
            ocr_text, ocr_err = _ocr_file(file_path)
            if ocr_err:
                return None, f"扫描件 PDF 需要 OCR，但: {ocr_err}"
            return ocr_text, None

        return text, None
    except ImportError:
        return None, "缺少 pymupdf，请运行: pip install pymupdf"
    except Exception as e:
        return None, f"PDF 解析失败: {str(e)}"


def _ocr_file(file_path):
    """
    用 PaddleOCR 识别图片或扫描件 PDF，返回 (text, error)
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from ocr import main as ocr_main
        result = ocr_main({"file_path": file_path, "engine": "auto"})
        result = json.loads(result)
        if result.get("status") == "success":
            return result.get("full_text", ""), None
        else:
            return None, result.get("message", "OCR 失败")
    except ImportError as e:
        return None, f"paddleocr 未安装: {e}（请运行 pip install paddleocr paddlepaddle）"
    except Exception as e:
        return None, f"OCR 执行异常: {e}"

def extract_txt(file_path):
    """提取 TXT 文本，返回 (text_or_None, error_or_None)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read(), None
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

    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    TEXT_EXTS  = {'.txt', '.md', '.markdown'}

    if ext == '.pdf':
        text, error = extract_pdf(file_path, auto_ocr=True)
        if error:
            return json.dumps({"status": "error", "message": error}, ensure_ascii=False)
    elif ext in TEXT_EXTS:
        text, error = extract_txt(file_path)
        if not text:
            return json.dumps({"status": "error", "message": "文本为空或读取失败"}, ensure_ascii=False)
    elif ext in IMAGE_EXTS:
        text, error = _ocr_file(file_path)
        if error:
            return json.dumps({"status": "error", "message": f"图片 OCR 失败: {error}"}, ensure_ascii=False)
    else:
        return json.dumps({
            "status": "error",
            "message": f"不支持的文件格式: {ext}，支持 .pdf, .txt, .md, .png, .jpg, .jpeg, .bmp, .tiff"
        }, ensure_ascii=False)
    
    # 生成文档 ID（基于文件路径的哈希）
    doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
    
    # 分块
    chunks = chunk_text(text, chunk_size=chunk_size)
    
    # 保存文本块到本地（供后续向量化使用）
    data_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    )
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
    parser = argparse.ArgumentParser(description="文档解析工具（支持 PDF/TXT/图片/扫描件）")
    parser.add_argument("file_path", help="文档路径")
    parser.add_argument("--chunk-size", type=int, default=500, help="分块大小")
    args = parser.parse_args()
    
    result = main({"file_path": args.file_path, "chunk_size": args.chunk_size})
    print(result)
