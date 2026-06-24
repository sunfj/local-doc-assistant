"""
工具4: OCR 文字识别 (ocr.py)
功能：从图片（PNG/JPG）或扫描件 PDF（无文字层）中提取文字
技术栈：RapidOCR（基于 PaddleOCR PP-OCRv4 ONNX 模型，无需 PaddlePaddle）

用法:
  python tools/ocr.py contract_scan.png
  python tools/ocr.py contract_scan.pdf --pages 1-3
  python tools/ocr.py image.jpg --engine openvino
"""
import os
import sys
import json
import argparse
from pathlib import Path

_OCR_ENGINE_CACHE = {"instance": None, "engine": None}


def _create_ocr_engine(engine="auto"):
    """
    创建 RapidOCR 实例：
    - onnxruntime：通用，pip install rapidocr-onnxruntime
    - openvino：性能最佳，pip install rapidocr-openvino
    - auto：优先 openvino，失败回退 onnxruntime
    """
    cache_key = engine
    if _OCR_ENGINE_CACHE["instance"] is not None and _OCR_ENGINE_CACHE["engine"] == cache_key:
        return _OCR_ENGINE_CACHE["instance"], None

    # 尝试 OpenVINO 后端
    if engine in ("auto", "openvino"):
        try:
            from rapidocr_openvino import RapidOCR
            ocr = RapidOCR()
            _OCR_ENGINE_CACHE["instance"] = ocr
            _OCR_ENGINE_CACHE["engine"] = cache_key
            return ocr, None
        except ImportError:
            if engine == "openvino":
                return None, "rapidocr-openvino 未安装，请运行: pip install rapidocr-openvino"

    # ONNX Runtime 后端（默认）
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        _OCR_ENGINE_CACHE["instance"] = ocr
        _OCR_ENGINE_CACHE["engine"] = cache_key
        return ocr, None
    except ImportError:
        return None, (
            "rapidocr 未安装，请运行: pip install rapidocr-onnxruntime\n"
            "（或性能更佳: pip install rapidocr-openvino）"
        )


def _ocr_image(ocr, image_path):
    """识别单张图片，返回 [{"text": str, "score": float, "bbox": list}, ...]"""
    result, elapse = ocr(str(image_path))
    lines = []
    if result is None:
        return lines
    for item in result:
        bbox, text, score = item
        lines.append({
            "text": text,
            "score": round(float(score), 4),
            "bbox": bbox,
        })
    return lines


def _ocr_pdf_scanned(ocr, pdf_path, pages=None):
    """
    对扫描件 PDF（无文字层）做 OCR：
    先用 PyMuPDF 逐页渲染为图片，再逐页 OCR。
    pages: 指定页码（1-based），如 "1-3" 或 "1,3,5" 或 None（全部）
    """
    try:
        import fitz
    except ImportError:
        return None, "缺少 pymupdf，请运行: pip install pymupdf"

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if pages is None:
        page_list = list(range(total_pages))
    else:
        page_list = _parse_page_range(pages, total_pages)

    all_lines = []
    import tempfile
    for pn in page_list:
        if pn < 0 or pn >= total_pages:
            continue
        page = doc[pn]
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        img = fitz.Pixmap(pix, 0) if pix.alpha else pix
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img.tobytes("png"))
            tmp_path = tmp.name
        try:
            lines = _ocr_image(ocr, tmp_path)
            for line in lines:
                line["page"] = pn + 1  # 1-based
            all_lines.extend(lines)
        finally:
            os.unlink(tmp_path)

    doc.close()
    return all_lines, None


def _parse_page_range(pages_str, total_pages):
    """解析 '1-3' 或 '1,3,5' 为 0-based 页码列表"""
    result = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a) - 1, int(b) - 1
            result.extend(range(a, min(b + 1, total_pages)))
        else:
            pn = int(part) - 1
            if 0 <= pn < total_pages:
                result.append(pn)
    return sorted(set(result))


def _is_scanned_pdf(pdf_path):
    """判断 PDF 是否为扫描件（无文字层或文字极少）"""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        total_chars = sum(len(page.get_text()) for page in doc)
        doc.close()
        return total_chars < 50
    except Exception:
        return False


def main(args):
    """
    入口函数。
    args: {"file_path": "...", "pages": "1-3", "engine": "auto"}
    """
    file_path = args.get("file_path", "")
    if not file_path:
        return json.dumps({"status": "error", "message": "缺少 file_path"}, ensure_ascii=False)

    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        return json.dumps({"status": "error", "message": f"文件不存在: {file_path}"}, ensure_ascii=False)

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".pdf"):
        return json.dumps({
            "status": "error",
            "message": f"不支持的格式: {ext}，支持 .png/.jpg/.jpeg/.bmp/.tiff/.pdf"
        }, ensure_ascii=False)

    engine = args.get("engine", "auto")
    pages = args.get("pages", None)

    ocr, err = _create_ocr_engine(engine=engine)
    if err:
        return json.dumps({"status": "error", "message": err}, ensure_ascii=False)

    try:
        if ext == ".pdf":
            if not _is_scanned_pdf(file_path):
                return json.dumps({
                    "status": "error",
                    "message": "PDF 包含文字层，请使用 parse_document 代替 OCR",
                    "hint": "parse_document 可直接提取 PDF 文字，无需 OCR"
                }, ensure_ascii=False)
            lines, err = _ocr_pdf_scanned(ocr, file_path, pages=pages)
        else:
            lines = _ocr_image(ocr, file_path)

        if err:
            return json.dumps({"status": "error", "message": err}, ensure_ascii=False)

        full_text = "\n".join(line["text"] for line in lines)
        avg_score = (
            round(sum(line["score"] for line in lines) / len(lines), 4) if lines else 0
        )

        return json.dumps({
            "status": "success",
            "file": os.path.basename(file_path),
            "total_lines": len(lines),
            "avg_score": avg_score,
            "full_text": full_text,
            "lines": lines,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": f"OCR 执行失败: {e}"}, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地 OCR 文字识别工具（RapidOCR）")
    parser.add_argument("file_path", help="图片或扫描件 PDF 路径")
    parser.add_argument("--engine", default="auto", choices=["auto", "openvino", "onnxruntime"],
                        help="OCR 后端（auto 优先 openvino，失败回退 onnxruntime）")
    parser.add_argument("--pages", default=None, help="PDF 页码范围，如 1-3 或 1,3,5")
    cli_args = parser.parse_args()

    result = main({
        "file_path": cli_args.file_path,
        "engine": cli_args.engine,
        "pages": cli_args.pages,
    })
    print(result)
