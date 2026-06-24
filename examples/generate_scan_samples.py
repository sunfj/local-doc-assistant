"""
生成扫描件风格的 demo 素材：
- contract_scan.png   合同照片（带轻度旋转/噪点/纸张色，模拟手机拍照）
- contract_scan.pdf   扫描件 PDF（无文字层，纯图片）
- contract_clean.png  干净版（白底黑字）作为对比
用法：
  python examples/generate_scan_samples.py
"""
import os
import glob
import random
import io
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "examples")
os.makedirs(OUT_DIR, exist_ok=True)


def find_chinese_font(size):
    """找系统中文字体"""
    candidates = (
        glob.glob("/System/Library/Fonts/*.ttc")
        + glob.glob("/System/Library/Fonts/*.ttf")
        + glob.glob("/System/Library/Fonts/Supplemental/*.ttf")
        + glob.glob("/usr/share/fonts/**/*.ttc", recursive=True)
        + glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
        + glob.glob("/Library/Fonts/*.ttf")
    )
    for f in candidates:
        low = f.lower()
        if any(k in low for k in ("pingfang", "songti", "stheiti", "hiragino", "noto")):
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                continue
    # fallback
    return ImageFont.load_default()


# 合同正文（精简版 contract_sample.txt 用于版面排版）
CONTRACT_TEXT = """\
智能设备采购与服务合同

合同编号：AIPC-2026-0518
签订日期：2026年5月18日
签订地点：北京市海淀区

甲方（采购方）：北京智算科技有限公司
乙方（供货方）：英特尔（中国）有限公司

第一条 合同标的
甲方拟向乙方采购 200 台 Intel Core Ultra 商用笔记本电脑，每台单价
人民币 12,800 元，合同总价为人民币 256 万元整。

第二条 交付时间
乙方应于 2026 年 6 月 30 日前将全部设备送达甲方指定仓库。

第五条 违约责任
5.1 乙方逾期交付的，每逾期一日，应向甲方支付合同总价 0.5% 的违约金，
    违约金累计不超过合同总价的 10%。
5.2 乙方交付的设备不符合质量标准的，甲方有权解除合同，乙方应退还已收
    全部款项，并额外支付合同总价 15% 作为赔偿金。
5.3 甲方逾期付款的，每逾期一日，应向乙方支付应付款项 0.3% 的滞纳金，
    滞纳金累计不超过应付款项的 5%。

第六条 保修条款
乙方提供 36 个月整机保修服务。NPU 模块单独提供 48 个月延长保修。

甲方（盖章）：北京智算科技有限公司
乙方（盖章）：英特尔（中国）有限公司
"""


def render_contract(size=(900, 1400), font_size=20, line_height=30, margin=60):
    """把合同文本渲染到白底图片上，返回 PIL.Image"""
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = find_chinese_font(font_size)
    y = margin
    for line in CONTRACT_TEXT.splitlines():
        d.text((margin, y), line, fill=(20, 20, 20), font=font)
        y += line_height
    return img


def apply_scan_effects(img):
    """模拟扫描/拍照效果：纸张泛黄 + 轻度旋转 + 噪点 + 模糊"""
    # 1) 纸张色：白色 → 米色
    paper = Image.new("RGB", img.size, (250, 246, 235))
    img = Image.blend(paper, img, 0.92)

    # 2) 轻度高斯模糊（模拟拍照失焦）
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    # 3) 加椒盐噪点
    px = img.load()
    rng = random.Random(42)
    for _ in range(int(img.size[0] * img.size[1] * 0.003)):
        x = rng.randint(0, img.size[0] - 1)
        y = rng.randint(0, img.size[1] - 1)
        c = rng.randint(180, 230)
        px[x, y] = (c, c, c)

    # 4) 轻度旋转（拍照常见，± 1.5°）
    img = img.rotate(-1.3, resample=Image.BICUBIC, fillcolor=(250, 246, 235))

    return img


def save_pdf_image_only(img, out_pdf):
    """把图片保存为单页 PDF（纯图片，无文字层）"""
    img.convert("RGB").save(out_pdf, "PDF", resolution=150.0)


def main():
    print("生成合同 demo 素材...")
    # 1) 干净版 PNG（对照）
    clean = render_contract()
    clean_path = os.path.join(OUT_DIR, "contract_clean.png")
    clean.save(clean_path, "PNG")
    print(f"  ✓ {clean_path}  ({os.path.getsize(clean_path)//1024} KB)")

    # 2) 扫描风格 PNG
    scan = apply_scan_effects(render_contract())
    scan_path = os.path.join(OUT_DIR, "contract_scan.png")
    scan.save(scan_path, "PNG")
    print(f"  ✓ {scan_path}  ({os.path.getsize(scan_path)//1024} KB)")

    # 3) 扫描风格 PDF（图片版，无文字层）
    pdf_path = os.path.join(OUT_DIR, "contract_scan.pdf")
    save_pdf_image_only(scan, pdf_path)
    print(f"  ✓ {pdf_path}  ({os.path.getsize(pdf_path)//1024} KB)")

    # 4) 验证 PDF 是否真的无文字层（用 fitz 提取应该返回空）
    try:
        import fitz
        doc = fitz.open(pdf_path)
        extracted = "".join(p.get_text() for p in doc)
        doc.close()
        if extracted.strip():
            print(f"  ⚠ PDF 含文字层 {len(extracted)} 字 —— 不是纯扫描件")
        else:
            print(f"  ✓ PDF 无文字层 —— 真正的扫描件，PyMuPDF 提取不到任何文字")
    except Exception as e:
        print(f"  (验证 PDF 时出错: {e})")

    print("\n生成完成。")


if __name__ == "__main__":
    main()
