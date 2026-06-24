# local-doc-assistant — AI PC 本地文档助手

> 基于 OpenVINO INT4 + PaddleOCR 的私有化 OCR + RAG Agent Skill。
> 扫描合同、拍照证件、电子文档——全程本地识别与问答，数据不出机。

**魔搭 Skills 中心**：[https://modelscope.cn/skills/seunal/local-doc-assistant](https://modelscope.cn/skills/seunal/local-doc-assistant)

---

## 核心能力

- **本地 OCR 文字识别**：扫描件 PDF / 拍照图片 → PaddleOCR PP-OCRv6 本地识别，中英文置信度 98%+
- **本地文档解析**：PDF（文字版+扫描件）/ TXT / Markdown / 图片，自动分块并建立索引
- **语义检索**：基于 OpenVINO BGE INT4 Embedding + FAISS 向量检索
- **智能问答**：通过 Ollama/Qwen3.5 等 ≤35B 模型，基于文档内容生成准确回答
- **隐私保护**：全流程本地完成，数据不出机，适合处理合同、病历、身份证等敏感文档

---

## 快速开始

### 一键初始化（推荐）

```bash
bash setup.sh   # 安装依赖 + 下载 BGE 模型 + 验证链路
```

### 手动安装

```bash
python -m pip install -r requirements.txt

# 导出 BGE 模型（约 2-5 分钟，需网络）
python export_models.py --model bge
```

### 运行 Agent

```bash
# 确保 Ollama 已安装并运行
ollama pull qwen3.5:9b

# 电子文档问答
python run_agent.py "请解析 examples/sample.txt 并告诉我 OpenVINO 支持哪些硬件设备？" \
  --backend ollama --model qwen3.5:9b

# 扫描件合同问答（图片直接 OCR → RAG 问答）
python run_agent.py "请解析 examples/contract_scan.pdf，违约金怎么算？" \
  --backend ollama --model qwen3.5:9b
```

---

## 支持的文档格式

| 格式 | 处理方式 | 说明 |
|------|----------|------|
| **PDF（文字版）** | PyMuPDF 直接提取 | 电子 PDF，含文字层 |
| **PDF（扫描件）** | 自动检测 → PaddleOCR | 无文字层的扫描件/照片 PDF |
| **图片** | PaddleOCR 直接识别 | PNG / JPG / BMP / TIFF |
| **TXT / Markdown** | 直接读取 | UTF-8 纯文本 |

**扫描件自动路由**：`parse_document` 会自动判断 PDF 是否有文字层——无文字层时自动调用 OCR，无需用户手动切换。

---

## 验证测试

```bash
# 全量单元测试（9 个用例，含 3 个 OCR 测试）
pytest tests/test_skill.py -v

# 烟雾测试（不依赖 Ollama）
python examples/smoke_test.py

# 生成扫描件 demo 素材
python examples/generate_scan_samples.py

# 单独测试 OCR
python tools/ocr.py examples/contract_scan.png
```

---

## 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 内存 | 8GB | 16GB |
| 处理器 | Intel 6th Gen+ / Apple M1+ | Intel Core Ultra |
| 加速 | CPU | GPU / NPU |

---

## 性能数据

| 操作 | CPU (Mac ARM) | Core Ultra (NPU) | Arc GPU |
|------|---------------|-------------------|---------|
| OCR 单页扫描件 | ~3s | ~1.5s | ~1s |
| 文档解析 (100页 PDF) | 2.3s | 2.3s | 2.3s |
| 向量构建 (1000 块) | 8.5s | 5.2s | 3.1s |
| RAG 查询 | 1.8s | 1.2s | 0.9s |
| LLM 推理 (INT4) | 30 tok/s | 45 tok/s | 80 tok/s |

---

## 技术栈

- **OCR**：PaddleOCR PP-OCRv6_medium（PaddlePaddle CPU / OpenVINO 后端）
- **LLM**：Qwen3.5-9B（Ollama）或 Qwen2.5-7B INT4（OpenVINO）
- **Embedding**：BAAI/bge-small-zh-v1.5 INT4（约 19MB）
- **向量库**：FAISS-CPU
- **推理框架**：OpenVINO 2026.x + PaddlePaddle 3.x
- **量化**：NNCF INT4，group_size=64

---

## 目录结构

```
local-doc-assistant/
├── manifest.json          # Skill 元数据 + 4 个工具定义
├── SKILL.md               # QwenPaw / Anthropic 风格 LLM 工作流指令
├── setup.sh               # 一键初始化脚本
├── requirements.txt       # 依赖
├── README.md              # 本文件
├── ARTICLE.md             # 参赛文章
├── export_models.py       # BGE/Qwen 模型导出脚本
├── run_agent.py           # Ollama/OpenVINO 双后端 Agent
├── tools/
│   ├── ocr.py             # Tool 1: 本地 OCR（PaddleOCR）
│   ├── doc_parser.py      # Tool 2: 文档解析（自动路由 OCR）
│   ├── vector_store.py    # Tool 3: 向量化索引（OpenVINO BGE）
│   ├── rag_query.py       # Tool 4: 语义检索（FAISS）
│   ├── embedding.py       # 共享 Embedding 模块
│   └── llm.py             # OpenVINO LLM 推理
├── examples/
│   ├── sample.txt             # 技术文档示例
│   ├── contract_sample.txt    # 合同文本示例
│   ├── whitepaper_sample.pdf  # PDF 白皮书示例
│   ├── contract_scan.png      # 扫描风格合同照片
│   ├── contract_scan.pdf      # 无文字层扫描件 PDF
│   ├── contract_clean.png     # 干净版对照
│   ├── generate_scan_samples.py # 素材生成脚本
│   └── smoke_test.py          # 链路验证
├── tests/
│   └── test_skill.py      # 单元测试（9 个用例）
├── data/                  # 运行时生成（索引/向量）
└── models/                # 导出的 OpenVINO 模型
```

---

## 创新点

1. **扫描件自动检测 + OCR 路由**：`parse_document` 自动判断 PDF 是否有文字层，无文字层时无缝切换到 PaddleOCR，用户无需关心底层差异
2. **合同照片 → 智能问答**：拍照/扫描的纸质文档，本地 OCR 识别 → 向量索引 → 自然语言问答，全流程数据不出机
3. **共享 Embedding 模块**：4 个 Tool 共享同一份 OpenVINO BGE 模型，避免重复加载（节省 800ms/次）
4. **INT4 + L2 归一化 + IndexFlatIP**：量化精度与语义检索在 FAISS 上一次性对齐
5. **框架无关**：manifest.json 转 OpenAI schema 后，Ollama / QwenPaw / LangChain 全部兼容
6. **硬件无关**：embedding/LLM 通过 `device` 参数切换 CPU/GPU/NPU，无需改业务代码

---

## 典型场景

### 扫描合同智能问答
```
用户：（上传合同扫描件照片）
      这份合同的违约金条款是什么？
→ Skill：本地 OCR 识别（22 行，置信度 98%+）→ 向量索引 → 问答
→ 回答：每逾期一日，支付合同总价 0.5% 的违约金，累计不超过 10%
```

### 扫描件 PDF 无文字层
```
用户：请解析 contract_scan.pdf，甲方是谁？
→ Skill：自动检测无文字层 → 调用 OCR → 识别出全文 → 索引 → 检索
→ 回答：甲方是北京智算科技有限公司
```

---

## 许可证

Apache 2.0
