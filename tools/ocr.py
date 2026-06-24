"""
工具4: OCR 文字识别 (ocr.py)
功能：从图片（PNG/JPG）或扫描件 PDF（无文字层）中提取文字
技术栈：PaddleOCR (PP-OCRv6_medium) / PaddlePaddle 或 OpenVINO 后端

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

# 尝试逐级加载 OCR 引擎：OpenVINO > PaddlePaddle > 失败
_OCR_ENGINE_CACHE = {"instance": None, "engine": None}


def _create_ocr_engine(engine="auto", lang="ch"):
    cache_key = (engine, lang)
    if _OCR_ENGINE_CACHE["instance"] is not None and _OCR_ENGINE_CACHE["engine"] == cache_key:
        return _OCR_ENGINE_CACHE["instance"], None

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return None, "paddleocr 未安装，请运行: pip install paddleocr paddlepaddle"

    kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "lang": lang,
    }

    # 尝试 OpenVINO 后端
    if engine in ("auto", "openvino"):
        try:
            ocr = PaddleOCR(enable_hpi=True, **kwargs)
            _OCR_ENGINE_CACHE["instance"] = ocr
            _OCR_ENGINE_CACHE["engine"] = cache_key
            return ocr, None
        except Exception:
            if engine == "openvino":
                return None, "OpenVINO 后端不可用，请先运行: paddleocr install_hpi_deps cpu"

    # PaddlePaddle 后端（默认）
    try:
        ocr = PaddleOCR(**kwargs)
        _OCR_ENGINE_CACHE["instance"] = ocr
        _OCR_ENGINE_CACHE["engine"] = cache_key
        return ocr, None
    except Exception as e:
        return None, f"OCR 引擎初始化失败: {e}"


def _ocr_image(ocr, image_path):
    """识别单张图片，返回 [(text, score), ...]"""
    result = ocr.predict(str(image_path))
    lines = []
    for page in result:
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        boxes = page.get("rec_boxes", [])
        for i, (t, s) in enumerate(zip(texts, scores)):
            line = {"text": t, "score": round(float(s), 4)}
            if i < len(boxes) and boxes[i] is not None:
                line["bbox"] = boxes[i].tolist() if hasattr(boxes[i], "tolist") else boxes[i]
            lines.append(line)
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

    # 解析页码范围
    if pages is None:
        page_list = list(range(total_pages))
    else:
        page_list = _parse_page_range(pages, total_pages)

    all_lines = []
    for pn in page_list:
        if pn < 0 or pn >= total_pages:
            continue
        page = doc[pn]
        # 渲染为高分辨率图片（300 DPI）
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        img = fitz.Pixmap(pix, 0) if pix.alpha else pix  # 去 alpha
        # 转为 bytes 给 PaddleOCR
        import tempfile
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
        # 每页平均不到 10 个字符 → 视为扫描件
        return total_chars < 50
    except Exception:
        return False


def _render_pdf_page_to_image(pdf_path, page_number=0):
    """将 PDF 某页渲染为临时图片路径（供 OCR 使用）"""
    import fitz
    doc = fitz.open(str(pdf_path))
    if page_number < 0 or page_number >= len(doc):
        doc.close()
        return None, f"页码 {page_number + 1} 超出范围（共 {len(doc)} 页）"
    page = doc[page_number]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(pix.tobytes("png"))
        tmp_path = tmp.name
    doc.close()
    return tmp_path, None


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
    lang = args.get("lang", "ch")
    pages = args.get("pages", None)

    ocr, err = _create_ocr_engine(engine=engine, lang=lang)
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
    parser = argparse.ArgumentParser(description="本地 OCR 文字识别工具（PaddleOCR）")
    parser.add_argument("file_path", help="图片或扫描件 PDF 路径")
    parser.add_argument("--engine", default="auto", choices=["auto", "openvino", "paddle"],
                        help="OCR 后端（auto 优先 openvino，失败回退 paddle）")
    parser.add_argument("--lang", default="ch", help="识别语言（ch=中文，en=英文）")
    parser.add_argument("--pages", default=None, help="PDF 页码范围，如 1-3 或 1,3,5")
    cli_args = parser.parse_args()

    result = main({
        "file_path": cli_args.file_path,
        "engine": cli_args.engine,
        "lang": cli_args.lang,
        "pages": cli_args.pages,
    })
    print(result)
