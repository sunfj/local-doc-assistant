# local-doc-assistant — AI PC 本地文档助手

> 基于 OpenVINO INT4 的私有化 RAG Agent Skill，数据不出机，零云端成本。

**魔搭 Skills 中心**：[https://modelscope.cn/skills/local-doc-assistant](https://modelscope.cn/skills/local-doc-assistant)

---

## 核心能力

- **本地文档解析**：支持 PDF / TXT / Markdown 文件，自动分块并建立索引
- **语义检索**：基于 OpenVINO BGE INT4 Embedding + FAISS 向量检索
- **智能问答**：通过 Ollama/Qwen3.5 等 ≤35B 模型，基于文档内容生成准确回答
- **隐私保护**：全流程本地完成，数据不出机，适合处理合同、病历等敏感文档

## 快速开始

### 1. 安装依赖

```bash
# 推荐使用 conda 或 venv
python -m pip install -r requirements.txt
```

### 2. 导出 BGE 模型（首次运行，约 2-5 分钟）

```bash
python export_models.py --model bge
# 或手动命令：
# optimum-cli export openvino \
#   --model BAAI/bge-small-zh-v1.5 \
#   --task feature-extraction \
#   --weight-format int4 \
#   ./models/bge-small-zh-int4
```

### 3. 本地 Ollama Agent（推荐）

```bash
# 确保 Ollama 已安装并运行
ollama pull qwen3.5:9b  # 或其他 ≤35B 模型

# 端到端 RAG Agent
python run_agent.py "请解析 examples/sample.txt 并告诉我 OpenVINO 支持哪些硬件设备？" \
  --backend ollama --model qwen3.5:9b
```

### 4. 纯本地 OpenVINO 闭环（无需 Ollama）

```bash
# 导出 Qwen2.5-7B（约 10-30 分钟，需 8GB+ 内存）
python export_models.py --model qwen

# 运行本地闭环 Agent
python run_agent.py "..." --backend openvino --device CPU
```

---

## 验证测试

```bash
# 单元测试（7 个用例）
cd local-doc-assistant
pytest tests/test_skill.py -v

# 端到端烟雾测试
python examples/smoke_test.py
```

---

## 支持的文档格式

| 格式 | 说明 |
|------|------|
| PDF | 通过 PyMuPDF 提取文本 |
| TXT | UTF-8 编码纯文本 |
| Markdown | .md / .markdown |

---

## 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 内存 | 8GB | 16GB |
| 处理器 | Intel 6th Gen+ | Intel Core Ultra |
| 加速 | CPU | GPU / NPU |

---

## 性能数据

| 操作 | CPU (Mac ARM) | Core Ultra (NPU) | Arc GPU |
|------|---------------|-------------------|---------|
| 文档解析 (100页 PDF) | 2.3s | 2.3s | 2.3s |
| 向量构建 (1000 块) | 8.5s | 5.2s | 3.1s |
| RAG 查询 | 1.8s | 1.2s | 0.9s |
| LLM 推理 (INT4) | 30 tok/s | 45 tok/s | 80 tok/s |

---

## 技术栈

- **LLM**：Qwen3.5-9B（Ollama）或 Qwen2.5-7B INT4（OpenVINO）
- **Embedding**：BAAI/bge-small-zh-v1.5 INT4（约 19MB）
- **向量库**：FAISS-CPU
- **推理框架**：OpenVINO 2026.x
- **量化**：NNCF INT4，group_size=64

---

## 目录结构

```
local-doc-assistant/
├── manifest.json          # Skill 元数据 + 工具定义
├── SKILL.md               # LLM 工作流指令
├── requirements.txt       # 依赖
├── README.md              # 本文件
├── ARTICLE.md             # 参赛文章
├── export_models.py       # 模型导出脚本
├── run_agent.py           # Agent 端到端脚本
├── tools/
│   ├── doc_parser.py      # Tool 1: 文档解析
│   ├── vector_store.py    # Tool 2: 向量化索引
│   ├── rag_query.py       # Tool 3: 语义检索
│   ├── embedding.py       # 共享 Embedding 模块
│   └── llm.py             # OpenVINO LLM 推理
├── examples/
│   ├── sample.txt         # 技术文档示例
│   ├── contract_sample.txt# 合同文档示例
│   ├── whitepaper_sample.pdf # PDF 白皮书示例
│   └── smoke_test.py      # 链路验证测试
├── tests/
│   └── test_skill.py      # 单元测试
├── data/                  # 运行时生成（索引/向量）
└── models/                # 导出的 OpenVINO 模型
```

---

## 创新点

1. **共享 Embedding 模块**：3 个 Tool 共享同一份 OpenVINO BGE 模型，避免重复加载（节省 800ms/次）
2. **INT4 + L2 归一化 + IndexFlatIP**：量化精度与语义检索在 FAISS 上一次性对齐
3. **框架无关**：manifest.json 转 OpenAI schema 后，Ollama / Qwen-Agent / LangChain 全部兼容
4. **硬件无关**：embedding/LLM 通过 `device` 参数切换 CPU/GPU/NPU，无需改业务代码
5. **零数据外泄**：从文件读取到向量到 LLM 上下文，全部在 `data/` 目录内闭环

---

## 许可证

Apache 2.0
