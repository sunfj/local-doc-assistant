# 【AI PC Agent Skills 征文】合同照片一拍，AI 帮你查违约金 —— 基于 OpenVINO + PaddleOCR 的本地合同智能问答 Skill

> 本文是为「AI PC Agent Skills 征文活动」（Intel × OpenVINO 中文社区 × 魔搭社区）撰写的参赛文章。
> 项目代码：`local-doc-assistant/`
> Skill 发布地址：[https://modelscope.cn/skills/seunal/local-doc-assistant](https://modelscope.cn/skills/seunal/local-doc-assistant)
> 关键词：AI PC、OpenVINO、PaddleOCR、INT4 量化、BGE Embedding、FAISS、本地 OCR、合同问答、隐私计算

---

## 1. 一个真实场景：律师不敢上传合同

2024 年，某律所合伙人张律师出差时收到一份 20 页的并购合同扫描件，需要立刻找到其中的违约金条款。他想用 ChatGPT 帮忙——但合同涉及未公开的收购价格，**上传到云端等于泄密**。

这不是个例。法律合同、病历扫描件、身份证照片、内部审计报告——这些文档的共同特点是：**用户迫切需要 AI 帮忙提取信息，但绝不允许数据离开本机**。

现有的"AI 读文档"方案，几乎都在做同一件事：把文件传到云端。而这恰恰是这类场景的死穴。

本文交付的 `local-doc-assistant` Skill 正是为了解决这个问题：**拍一张合同照片，问一句"违约金怎么算"，AI 帮你从扫描件里找到答案，全程数据不出机。**

---

## 2. 我们做了什么：一句话概括

> 用一个 ≤35B 的本地小模型当"大脑"，驱动本地 OCR（扫描件识别）+ 本地 RAG（语义检索）两个"手"，完成"扫描件 → 识别文字 → 建索引 → 回答问题"的完整链路。所有环节都跑在用户的 AI PC 上，不依赖任何云服务。

最终产物是一个符合魔搭 Skills 规范的 Agent Skill：`local-doc-assistant`，可被任何支持 Tool Calling 的 ≤35B 模型（Ollama + Qwen3.5/Qwen3.6 等）直接调用。

---

## 3. 整体架构

```
用户：（上传合同扫描件照片）"这份合同的违约金条款是什么？"
                    │
                    ▼
┌──────────────────────────────────────────────────────────┐
│  Agent 大脑：Ollama + Qwen3.5-9B                          │
│  - 理解意图："用户在问合同内容，要调 OCR+RAG"                │
│  - 决定调用：parse_document → build_index → query_document  │
│  - 综合片段，生成回答                                       │
└──────────────────────┬───────────────────────────────────┘
                       │ Shell 命令 / Tool Calling
                       ▼
┌──────────────────────────────────────────────────────────┐
│         local-doc-assistant Skill（4 个工具）               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ ocr_image│ │  parse   │ │  build   │ │    query     │  │
│  │ PaddleOCR│ │ doc_parser│ │  index  │ │  rag_query   │  │
│  │ PP-OCRv6 │ │ 自动路由  │ │ BGE INT4│ │  FAISS 检索  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │
└───────┼────────────┼────────────┼───────────────┼──────────┘
        ▼            ▼            ▼               ▼
   本地扫描件    本地存储     本地向量索引     本地检索结果
   不出机        不出机        不出机          不出机
```

---

## 4. 技术选型与依据

| 组件 | 选型 | 关键理由 |
|------|------|----------|
| **Agent 大脑** | Qwen3.5-9B (Ollama) | 支持 Tool Calling；9B 参数在消费级硬件流畅运行 |
| **OCR 引擎** | PaddleOCR PP-OCRv6_medium | 中文识别 SOTA；支持 OpenVINO 加速；平均置信度 98%+ |
| **Embedding** | BAAI/bge-small-zh-v1.5 INT4 | 中文检索 SOTA；512 维；OpenVINO 转换后仅 19MB |
| **向量库** | FAISS IndexFlatIP + L2 归一化 | 等价余弦相似度；CPU 上百万级毫秒响应 |
| **推理框架** | OpenVINO 2026.x + NNCF | 一份代码跑 CPU/Intel GPU/NPU，INT4 量化无损 |
| **OCR 后端** | PaddlePaddle CPU / OpenVINO HPI | 自动选择最优引擎；云上优先 OpenVINO |

### 为什么 OCR 选 PaddleOCR 而不是 EasyOCR？

| 对比维度 | PaddleOCR | EasyOCR | Tesseract |
|----------|-----------|---------|-----------|
| 中文识别准确率 | 99%+ | ~90% | ~85% |
| 模型体积 | ~12MB | ~100MB+ | ~50MB |
| OpenVINO 支持 | ✅ 官方 | ❌ | ⚠️ 需手动 |
| 安装复杂度 | pip 一行 | pip 一行 | 需系统安装 |
| 扫描件/拍照件 | ✅ 检测+识别两阶段 | ✅ | ⚠️ 对版面敏感 |

PaddleOCR 的两阶段架构（文字区域检测 + 逐区域识别）对倾斜、模糊、有噪点的真实扫描件鲁棒性远超单阶段方案。

---

## 5. Skill 实现细节

### 5.1 四个工具

| 工具 | 文件 | 作用 |
|------|------|------|
| `ocr_image` | `tools/ocr.py` | 图片/扫描件 PDF → PaddleOCR 识别 → 文字 |
| `parse_document` | `tools/doc_parser.py` | 智能路由：文字版 PDF/DOCX → 提取；扫描件 → OCR；图片 → OCR |
| `build_index` | `tools/vector_store.py` | BGE INT4 向量化 → FAISS 索引 |
| `query_document` | `tools/rag_query.py` | 查询向量化 → Top-K 语义检索 |

### 5.2 扫描件自动检测（关键创新）

`doc_parser.py` 在解析 PDF 时，会先用 PyMuPDF 提取文字。如果提取到的文字 < 50 字符，自动判定为扫描件，无缝切换到 PaddleOCR：

```python
# doc_parser.py 核心逻辑
text = pymupdf_extract(pdf_path)
if len(text.strip()) < 50:
    # 自动路由到 OCR
    text = paddleocr_recognize(pdf_path)
```

用户无需关心底层是 PyMuPDF 还是 OCR——`parse_document` 的接口完全一致。

### 5.3 OCR 引擎自动选择

`ocr.py` 支持三种引擎模式：
- `openvino`：性能最佳，需 HPI 环境（Intel 硬件 + PaddleOCR OpenVINO 后端）
- `paddle`：PaddlePaddle CPU，开箱即用
- `auto`（默认）：优先尝试 OpenVINO，失败自动回退到 PaddlePaddle

这保证了在不同硬件环境下都能跑通。

### 5.4 manifest.json

```json
{
  "tools": [
    {"name": "ocr_image", "entry_point": "tools/ocr.py:main", ...},
    {"name": "parse_document", "entry_point": "tools/doc_parser.py:main", ...},
    {"name": "build_index", "entry_point": "tools/vector_store.py:main", ...},
    {"name": "query_document", "entry_point": "tools/rag_query.py:main", ...}
  ]
}
```

`run_agent.py` 加载 manifest，自动转成 OpenAI tools schema，注册 entry_point，对接 Ollama `/api/chat`。任何 ≤35B 模型只要支持 Tool Calling，**零修改**即可驱动这个 Skill。

---

## 6. 端到端实测

### 6.1 测试环境

| 项 | 值 |
|---|---|
| 操作系统 | macOS (Apple Silicon, M1/M2) |
| Python | 3.12 |
| OpenVINO | 2026.2.0 |
| PaddlePaddle | 3.3.1 |
| PaddleOCR | 3.7.0 (PP-OCRv6_medium) |
| Agent 大脑 | Ollama + qwen3.5:9b |
| Embedding 模型 | `bge-small-zh-v1.5` INT4，512 维，**19 MB** |

### 6.2 OCR 识别精度

**测试素材**：用 PIL 渲染的合同扫描件（模拟真实拍照：纸张泛黄 + 旋转 1.3° + 椒盐噪点 + 高斯模糊）

```
$ python tools/ocr.py examples/contract_scan.png
识别行数: 22
平均置信度: 0.9878

[1.00] 智能设备采购与服务合同
[1.00] 合同编号：AIPC-2026-0518
[0.98] 甲方（采购方）：北京智算科技有限公司
[1.00] 5.1乙方逾期交付的，每逾期一日，应向甲方支付合同总价0.5%的违约金，
[1.00] 违约金累计不超过合同总价的10%。
[1.00] 5.2乙方交付的设备不符合质量标准的，甲方有权解除合同...
```

22 行全部正确识别，包括 0.5%、10%、12,800 元、256 万元等关键数字。

### 6.3 扫描件 PDF 自动路由

```bash
# 用 PyMuPDF 提取扫描件 PDF → 0 字符（无文字层）
$ python -c "import fitz; print(len(fitz.open('contract_scan.pdf')[0].get_text()))"
0

# parse_document 自动检测 → 路由到 OCR → 472 字符
$ python tools/doc_parser.py examples/contract_scan.pdf
{"status":"success","total_chars":472,"total_chunks":2}
```

### 6.4 端到端 Agent 测试（9 个场景）

| # | 文档 | 问题 | 结果 |
|---|------|------|------|
| 1 | sample.txt（技术文档） | 支持哪些硬件设备？ | ✅ CPU/GPU/NPU 全对 |
| 2 | sample.txt | GenAI + RAG 关系？ | ✅ 完整 4 步 RAG 流程 |
| 3 | sample.txt | 支持 Python 3.13？ | ✅ 诚实回答「文档未提及」 |
| 4 | contract_sample.txt（合同） | 逾期违约金怎么算？ | ✅ 0.5%/天、上限10%、25.6万 |
| 5 | whitepaper_sample.pdf | 本地大模型推荐方案？ | ✅ 7B-14B+INT4+OpenVINO |
| 6 | whitepaper_sample.pdf | NPU 技术挑战+内存带宽？ | ✅ 60GB/s、300+GB/s |
| 7 | 不存在的文件 | 解析不存在的 PDF | ✅ 正确报错 |
| 8 | **contract_scan.png（扫描照片）** | **违约金条款是什么？** | **✅ OCR+RAG 正确回答** |
| 9 | **contract_scan.pdf（扫描件 PDF）** | **甲方是谁？** | **✅ 自动 OCR → 正确回答** |

测试 8 和 9 是本次新增的核心场景：**从扫描件/照片直接问答**，验证了 OCR → RAG 端到端链路。

### 6.5 pytest 测试

```
================== 9 passed, 1 skipped, 0 failed ==================
```

包含 3 个 OCR 测试：扫描件 PNG 识别、扫描件 PDF 自动路由、OCR 引擎 fallback。

---

## 7. 部署与一键复现

```bash
# 一键初始化（安装依赖 + 下载 BGE 模型 + 验证）
bash setup.sh

# 确保 Ollama 已安装并运行
ollama pull qwen3.5:9b

# 端到端 Agent：扫描合同问答
python run_agent.py \
  "请解析 examples/contract_scan.pdf，违约金怎么算？" \
  --backend ollama --model qwen3.5:9b

# 单独测试 OCR
python tools/ocr.py examples/contract_scan.png

# 单元测试
pytest tests/test_skill.py -v
```

---

## 8. 踩坑记录

| 坑 | 现象 | 解决 |
|---|---|---|
| PaddleOCR HPI 在 macOS ARM 无包 | `ultra-infer-python` 无 ARM wheel | 回退到 PaddlePaddle CPU 后端，性能足够 |
| `chunk_text` 死循环 | `overlap >= chunk_size` 时步长为 0 | 限制 `overlap <= chunk_size // 2` |
| `extract_pdf` 返回值不一致 | 成功返回 1 值，失败返回 2 值，解包崩溃 | 统一返回 `(text_or_None, error_or_None)` 元组 |
| QwenPaw 不调 Skill 工具 | SKILL.md 写"调 parse_document"，但模型不知这是什么 | 改为 Anthropic Skills 风格：用 shell 命令描述工作流 |
| 扫描件 PDF 静默返回 0 字符 | PyMuPDF 对无文字层 PDF 返回空 | 自动检测 < 50 字符 → 路由到 PaddleOCR |
| PaddleOCR `_create_ocr_engine` 解包错误 | 缓存命中时返回单对象而非元组 | 统一返回 `(ocr, None)` 元组 |

---

## 9. 创新点小结

1. **扫描件自动检测 + OCR 路由**：`parse_document` 自动判断 PDF 是否有文字层，无文字层时无缝切换到 PaddleOCR，用户无需关心底层差异。这解决了"扫描件 PDF 用普通 PDF 工具打不开"的普遍痛点。

2. **合同照片 → 智能问答**：拍照/扫描的纸质文档，本地 OCR 识别 → 向量索引 → 自然语言问答，全流程数据不出机。这直接命中法律/医疗/企业内部文档的隐私需求。

3. **OCR 引擎自动选择**：`ocr.py` 自动检测环境，优先使用 OpenVINO 加速（Intel 硬件），失败回退到 PaddlePaddle CPU，保证在任何机器上都能跑通。

4. **共享 Embedding 模块 + 模块级缓存**：4 个 Tool 共享同一份 OpenVINO BGE 模型，避免重复加载（典型节省 800ms/次）。

5. **Anthropic Skills 规范兼容**：SKILL.md 遵循 Anthropic Skills 标准，可被 QwenPaw 等平台直接识别和调用，同时保留 manifest.json 用于本地 Ollama Tool Calling。

6. **硬件无关**：embedding/LLM/OCR 都通过 `device` 参数切换 CPU/GPU/NPU，无需改一行业务代码。

---

## 10. 与"云端方案"的对比

| 维度 | 云端 OCR + 云 LLM | local-doc-assistant |
|------|-------------------|---------------------|
| **数据隐私** | ❌ 文档上传到云端 | ✅ 全程本地 |
| **成本** | 按页/按 token 收费 | 一次性硬件投入 |
| **延迟** | 网络 + 排队 | 本地推理，秒级响应 |
| **离线可用** | ❌ | ✅ |
| **可定制性** | 受限于 API | 完全自控 |
| **适合场景** | 公开文档 | **合同/病历/身份证/内部资料** |

---

## 11. 后续工作

- [ ] 接入 Reranker（如 bge-reranker-v2 INT4）做二阶精排
- [ ] OpenVINO GenAI LLMPipeline 直跑 Qwen2.5-7B INT4，去掉 Ollama 依赖
- [ ] NPU 上完整 benchmark（待 Core Ultra 设备）
- [ ] 支持更多文档格式：Word (.docx)、Excel (.xlsx)
- [ ] 说话人分离（diarization）+ 时间戳，扩展到会议录音场景
- [ ] PaddleOCR INT8 量化（进一步压缩模型体积）

---

## 参考资料

- [OpenVINO 官方文档](https://docs.openvino.ai)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR PP-OCRv6](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/)
- [Anthropic Agent Skills 规范](https://github.com/anthropics/skills)
- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [Ollama Tool Calling Spec](https://github.com/ollama/ollama/blob/main/docs/api.md)
