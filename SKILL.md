---
name: scan-and-ask
description: "Use this skill any time the user wants to analyze, query, summarize, or extract content from a local document file (.pdf / .txt / .md / .png / .jpg) or a scanned/photographed paper document. This includes: answering questions about a specific PDF/TXT/Markdown file, finding clauses in contracts, searching through long reports, summarizing whitepapers, extracting facts from technical documentation, or recognizing text from scanned documents and photos. Trigger whenever the user mentions a local file path ending in .pdf, .txt, .md, .markdown, .png, .jpg, .jpeg, .bmp, .tiff, OR uses words like 'document', 'contract', 'report', 'whitepaper', 'paper', 'manual', 'scan', 'photo', 'OCR', '文档', '合同', '报告', '白皮书', '手册', '扫描件', '识别', '提取文字' together with a file reference. The skill uses PaddleOCR (PP-OCRv6) for local OCR text recognition + OpenVINO INT4 BGE embedding + FAISS for fully local, privacy-preserving Retrieval-Augmented Generation (RAG). All data stays on the machine — never uploaded to the cloud."
version: "1.1.0"
author: "OpenClaw"
tags: [AIPC, RAG, OpenVINO, INT4, privacy, local-first, document-analysis, pdf, retrieval]
license: "Apache-2.0"
metadata:
  builtin_skill_version: "1.1"
---

> **Important:** All commands below assume you are running them **from this Skill's directory**.
> Run with: `cd {this_skill_dir} && python tools/...`
> Or use the `cwd` parameter of `execute_shell_command`.

# Local Document Assistant Skill

A privacy-first RAG (Retrieval-Augmented Generation) skill that runs **fully on-device**: documents never leave the local machine. Powered by OpenVINO INT4 quantized BGE embedding + FAISS vector search.

## Prerequisites

This skill needs these to work:

- **Python ≥ 3.10**
- Python packages: `openvino>=2025.3`, `faiss-cpu>=1.7.4`, `pymupdf>=1.23.0`, `numpy>=1.24.0`, `transformers>=4.40.0`, `optimum[openvino,nncf]`, `modelscope`
- **BGE INT4 OpenVINO model** at `models/bge-small-zh-int4/openvino_model.xml`

If any prerequisite is missing, **run the one-shot setup script first** (see "Initial Setup" below). Do NOT keep retrying tool calls when a prerequisite is missing — it will fail every time. Install first, then retry.

## Quick Reference

| Task | Command |
|------|---------|
| Initial setup (install deps + download models) | `bash setup.sh` |
| Check environment only | `bash setup.sh --check` |
| Parse a document into chunks | `python tools/doc_parser.py <file_path>` |
| OCR a scanned image/PDF directly | `python tools/ocr.py <file_path>` |
| Build vector index for a parsed doc | `python tools/vector_store.py <doc_id>` |
| Semantic search inside a doc | `python tools/rag_query.py <doc_id> "<question>"` |
| End-to-end smoke test | `python examples/smoke_test.py` |

All tools print a single line of JSON to stdout. Read `status` field first: `success` or `error`.

---

## Initial Setup (run ONCE per machine)

Before the very first use on this machine, set up the environment. Use the built-in `execute_shell_command` tool:

```bash
cd {this_skill_dir} && bash setup.sh
```

This will:
1. Install all Python dependencies via `pip`
2. Download `BAAI/bge-small-zh-v1.5` from **ModelScope** (fast in China; falls back to HuggingFace if needed)
3. Convert it to OpenVINO INT4 format (~19MB) and place it at `models/bge-small-zh-int4/`
4. Run the smoke test to verify the full chain

Setup takes 2–5 minutes on a normal machine. If you see `BGE 模型：✓ 已就绪` at the end, you're done. After that, never run setup again unless a dependency was uninstalled.

**Want to quickly check status without installing?** Run:
```bash
cd {this_skill_dir} && bash setup.sh --check
```

---

## Standard Workflow (use this for every user question)

The skill expects this **strict 3-step pipeline** for any document query:

### Step 1 — Parse the document

```bash
cd {this_skill_dir} && python tools/doc_parser.py "<absolute_or_relative_path_to_doc>"
```

Supports `.pdf`, `.txt`, `.md`, `.markdown`. Returns JSON with a `doc_id` field — **remember this id**, the next two steps need it.

Example output:
```json
{"status":"success","doc_id":"580eecf047e3","total_chunks":2,"file_name":"sample.txt"}
```

### Step 2 — Build the vector index (only first time per file)

```bash
cd {this_skill_dir} && python tools/vector_store.py <doc_id>
```

Uses the OpenVINO BGE model to embed every chunk and build a FAISS index. Cached on disk — repeating Step 2 on the same `doc_id` is fast and idempotent.

Example output:
```json
{"status":"success","doc_id":"580eecf047e3","total_vectors":2,"vector_dim":512}
```

### Step 3 — Semantic query

```bash
cd {this_skill_dir} && python tools/rag_query.py <doc_id> "<user_question>" --top-k 3
```

Returns the top-K most relevant chunks with cosine similarity scores. **Always use these chunks as the grounding context for your final natural-language answer to the user.**

Example output:
```json
{"status":"success","results":[
  {"rank":1,"score":0.7527,"chunk_index":0,"text":"OpenVINO 支持的硬件设备包括：CPU、GPU、NPU..."},
  ...
]}
```

---

## How to answer the user

After Step 3, **compose your final answer using only the retrieved chunks**:

- If `score >= 0.5` on the top hit → confident answer, cite the chunk text
- If all scores `< 0.4` → say "文档中未找到明确相关内容，可能这份文档没有讨论该话题"
- Never make up content that's not in the retrieved chunks
- Always respond in **Chinese**

---

## Common Scenarios & Trigger Examples

### "请解析 examples/sample.txt 并告诉我 OpenVINO 支持哪些硬件设备？"
→ Run Step 1 on `examples/sample.txt`, then Step 2, then Step 3 with query="OpenVINO 支持哪些硬件设备?". Use the top chunks to answer.

### "examples/contract_sample.txt 里如果乙方逾期交货违约金怎么算？"
→ Same 3-step pipeline. Use `top-k=3` to make sure you catch all relevant clauses.

### "examples/whitepaper.pdf AI PC 部署本地大模型推荐什么方案？"
→ The skill handles PDFs natively via PyMuPDF — same 3-step flow. Do NOT use built-in `pdf` skill or `read_file` on PDFs; this skill's `parse_document` is purpose-built for this.

### "这张合同照片里的违约金是多少？"（附 .png/.jpg 文件）
→ `parse_document` handles images natively via PaddleOCR — same 3-step flow:
1. `python tools/ocr.py <image_path>` (or let `parse_document` auto-route)
2. Build index
3. Query
Do NOT use built-in `read_file` on images — it can't do OCR. This skill's `parse_document` will auto-detect images and call OCR.

### "examples/contract_scan.pdf 这份扫描件里甲方是谁？"
→ The skill detects scanned PDFs (no text layer) and auto-routes to PaddleOCR. Same 3-step pipeline. `parse_document` will show `total_chars > 0` after OCR succeeds.

### Multiple questions on the same document
→ Run Step 1 + Step 2 once (record the `doc_id`). For every subsequent question about the same file, **only run Step 3** with that `doc_id`. This is the main performance win — embedding is the expensive part.

---

## Error Handling

| Error message | Fix |
|---------------|-----|
| `Embedding 模型不存在` | Run `bash setup.sh` to download + convert BGE |
| `OpenVINO 未安装` / `faiss 缺失` | Run `bash setup.sh` (or `pip install -r requirements.txt`) |
| `文档未解析，请先调用 parse_document` | The `doc_id` you passed has no chunks file — re-run Step 1 |
| `向量索引不存在，请先调用 build_index` | Re-run Step 2 |
| `文件不存在` | Check the path; this skill does NOT search filesystem — user must give a valid path |
| `不支持的文件格式` | Only `.pdf`, `.txt`, `.md`, `.markdown`. Other formats: tell user to convert first |

---

## Anti-patterns (DON'T do these)

- ❌ Do NOT use the built-in `read_file` tool to open a PDF or image — it can't extract text. Use this skill's `parse_document` instead.
- ❌ Do NOT use the built-in `pdf` skill on scanned PDFs — it can't do OCR. This skill auto-detects scanned PDFs and calls PaddleOCR.
- ❌ Do NOT use the built-in `search` tool to find content inside a known file — use this skill's RAG pipeline for better semantic recall.
- ❌ Do NOT skip Step 1 or Step 2 and jump straight to Step 3 — `rag_query.py` needs a built index, it will error.
- ❌ Do NOT call `parse_document` twice on the same file in the same session — the `doc_id` is deterministic (md5 of file path), so reuse the earlier result.
- ❌ Do NOT install dependencies via `pip install` one-by-one — run `bash setup.sh` instead for a consistent environment.

---

## Why this skill matters (for your context awareness)

This skill exists because users in privacy-sensitive scenarios (legal contracts, medical records, internal company docs, scanned IDs) cannot upload files to cloud LLMs or cloud OCR services. The full pipeline — document parsing (including scanned PDF and photo OCR via PaddleOCR), vectorization (OpenVINO BGE INT4), semantic retrieval (FAISS), and final LLM answer composition — happens **entirely on the user's machine** using INT4-quantized models that fit easily in 8GB RAM. OpenVINO provides hardware acceleration on Intel CPU / GPU / NPU.

When a user asks about a local document — whether it's a text PDF, a scanned contract photo, or a handwritten note — this is almost always the right tool. Trigger eagerly, not conservatively.
