# 合同照片一拍，AI 帮你查违约金 —— 从零打造本地扫描文档智能问答 Skill 的实践之路

> **摘要**：本文记录了 `scan-and-ask` Skill 从需求分析、技术选型、多轮优化到最终在 QwenPaw 云端验证通过的完整实践路径。重点分享 OCR 引擎替换、硬件自适应、镜像源策略等优化心得，以及对 Hybrid AI（端云协同）的思考。
>
> 项目代码：[github.com/sunfj/local-doc-assistant](https://github.com/sunfj/local-doc-assistant)
>
> 关键词：AI PC、OpenVINO、RapidOCR、INT4 量化、BGE Embedding、FAISS、本地 OCR、Hybrid AI

---

## 一、问题的起点：一个律师的困境

2024 年，某律所合伙人张律师出差时收到一份 20 页的并购合同扫描件，需要立刻找到其中的违约金条款。他想用 ChatGPT 帮忙——但合同涉及未公开的收购价格，**上传到云端等于泄密**。

这不是个例。法律合同、病历扫描件、身份证照片、内部审计报告——这些文档的共同特点是：**用户迫切需要 AI 帮忙提取信息，但绝不允许数据离开本机**。

现有的"AI 读文档"方案，几乎都在做同一件事：把文件传到云端。而这恰恰是这类场景的死穴。

**我们需要的是一种"数据不出机"的智能文档问答方案——而且要能处理扫描件和照片，因为现实中大量文档并不是电子版。**

---

## 二、我们要做什么

一句话概括：

> 用一个 ≤35B 的本地小模型当"大脑"，驱动本地 OCR（扫描件识别）+ 本地 RAG（语义检索）两个"手"，完成 **"扫描件 → 识别文字 → 建索引 → 回答问题"** 的完整链路。所有环节都跑在用户的 AI PC 上，不依赖任何云服务。

最终产物是一个符合魔搭 Skills 规范的 Agent Skill：`scan-and-ask`，可被任何支持 Tool Calling 的模型直接调用。

---

## 三、整体架构

```
用户：（上传合同扫描件照片）"这份合同的违约金条款是什么？"
                    │
                    ▼
┌──────────────────────────────────────────────────────────┐
│  Agent 大脑：Qwen3.5-9B                                   │
│  - 理解意图："用户在问合同内容，要调 OCR+RAG"                │
│  - 决定调用：parse_document → build_index → query_document  │
│  - 综合片段，生成回答                                       │
└──────────────────────┬───────────────────────────────────┘
                       │ Shell 命令 / Tool Calling
                       ▼
┌──────────────────────────────────────────────────────────┐
│           scan-and-ask Skill（4 个工具）                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ ocr_image│ │  parse   │ │  build   │ │    query     │ │
│  │ RapidOCR │ │doc_parser│ │  index   │ │  rag_query   │ │
│  │ PP-OCRv4 │ │ 自动路由  │ │BGE INT4 │ │  FAISS 检索  │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘ │
└───────┼────────────┼────────────┼───────────────┼─────────┘
        ▼            ▼            ▼               ▼
   本地扫描件    本地存储     本地向量索引     本地检索结果
   不出机        不出机        不出机          不出机
```

---

## 四、实践路径：从想法到可运行的 Skill

### 4.1 第一步：跑通 RAG 基础链路

最初的原型非常简单——只有纯文本的 RAG：

```
TXT 文件 → 分块 → BGE Embedding → FAISS 索引 → 语义检索 → LLM 回答
```

技术栈选型：

| 组件 | 选型 | 理由 |
|------|------|------|
| Agent 大脑 | Qwen3.5-9B (Ollama) | 支持 Tool Calling；9B 参数在消费级硬件流畅运行 |
| Embedding | BAAI/bge-small-zh-v1.5 INT4 | 中文检索 SOTA；OpenVINO 转换后仅 19MB |
| 向量库 | FAISS IndexFlatIP + L2 归一化 | 等价余弦相似度；CPU 上百万级毫秒响应 |
| 推理框架 | OpenVINO 2026.x + NNCF | 一份代码跑 CPU/GPU/NPU，INT4 量化精度几乎无损 |

这一步的关键决策是 **用 OpenVINO 把 BGE 模型量化到 INT4**。原始 FP32 模型约 90MB，量化后只有 19MB，推理速度提升 2-3 倍，精度损失不到 1%。对于一个 300 万参数的小模型，INT4 量化的性价比极高。

### 4.2 第二步：加入 OCR 能力

纯文本 RAG 只能处理电子文档。但现实中，大量文档是扫描件或手机照片——PyMuPDF 直接提取只能拿到 0 个字符。

于是我们加入了 OCR 引擎。**这是整个项目最关键的扩展**，因为它让 Skill 从"只能读电子文档"升级为"能读任何文档"。

核心设计是 **扫描件自动检测 + 无缝路由**：

```python
# doc_parser.py 核心逻辑
text = pymupdf_extract(pdf_path)
if len(text.strip()) < 50:
    # 自动路由到 OCR
    text = ocr_recognize(pdf_path)
```

用户无需关心底层是 PyMuPDF 还是 OCR——`parse_document` 的接口完全一致。

### 4.3 第三步：适配 QwenPaw 云端平台

当 Skill 在本地 Ollama 上跑通后，我们尝试把它部署到 QwenPaw（一个类似 OpenClaw 的云端 AI 平台）。

**踩的第一个大坑**：QwenPaw 遵循 Anthropic Skills 规范，Skill 是"知识指南"而非"可执行程序"。它通过 `execute_shell_command` 调用 shell 命令，而不是通过 function calling 调用 Python 函数。

我们最初写的 SKILL.md 是这样的：

```markdown
# 错误写法
当用户需要解析文档时，调用 `parse_document` 工具。
```

QwenPaw 完全不理解 `parse_document` 是什么。正确的写法应该是：

```markdown
# 正确写法
当用户需要解析文档时，执行：
cd {skill_dir} && python tools/doc_parser.py <文件路径>
```

这次踩坑让我们深刻理解了 **Skill 规范的本质：它是给 AI 看的"操作手册"，不是给机器执行的"配置文件"**。

---

## 五、优化心得：三次关键决策

### 5.1 OCR 引擎替换：从 PaddleOCR 到 RapidOCR

**问题**：PaddleOCR 的安装依赖 PaddlePaddle 框架，完整安装需要 **500MB+**。在 QwenPaw 云端环境中，这个体积太大了，安装时间超过 10 分钟。

**调研**：

| OCR 方案 | 安装体积 | 中文精度 | 推理速度 | 架构 |
|----------|----------|----------|----------|------|
| PaddleOCR + PaddlePaddle | ~500MB | 98%+ | 快 | PP-OCRv4/v6 |
| RapidOCR (ONNX) | **~30MB** | **98%+** | **快** | PP-OCRv4 ONNX |
| EasyOCR | ~100MB+ | ~90% | 中 | CRAFT + CRNN |
| Tesseract | ~50MB | ~85% | 慢 | LSTM |

**发现**：RapidOCR 使用的是和 PaddleOCR **完全相同的 PP-OCRv4 模型**，只是把推理后端从 PaddlePaddle 换成了 ONNX Runtime。模型一样，精度一样，但安装体积从 500MB 降到 30MB——**降低了 94%**。

**决策**：用 RapidOCR 替代 PaddleOCR。

**效果**：
- 安装时间：10+ 分钟 → **1 分钟内**
- 模型精度：无变化（同一 PP-OCRv4 模型）
- 测试用例：10 个全部通过，含 3 个 OCR 专项测试

**教训**：不要迷信"官方实现"。在 ONNX 生态成熟的今天，用 ONNX Runtime 替代深度学习框架做推理，是降低部署成本的有效手段。

### 5.2 硬件适配：从手动指定到 AUTO

**问题**：最初的代码里，设备参数写死为 `device="CPU"`：

```python
# 旧代码
model = load_embedding_model(model_dir, device="CPU")
```

这意味着在 Intel Core Ultra 的 NPU 上跑，也不会自动切换——用户必须手动改代码。

**解决**：OpenVINO 支持 `device="AUTO"`，会自动检测并选择最优硬件：

```
NPU > GPU > CPU（按优先级自动选择）
```

一行代码改动，让同一份代码在 Mac（CPU）、Intel Core Ultra（NPU）、Arc GPU（独显）上都能自动适配，**零配置**。

```python
# 新代码
model = load_embedding_model(model_dir, device="AUTO")
```

**教训**：**默认值决定用户体验**。把 `CPU` 改成 `AUTO` 只是一个字符串的事，但它决定了用户是"开箱即用"还是"还得改配置"。

### 5.3 镜像源策略：从单源到多源 fallback

**问题**：在 QwenPaw 云端环境中，使用阿里云 pip 镜像安装 `openvino` 时报错"找不到匹配版本"。

**排查**：用 curl 直接检查镜像源，发现阿里云和清华**都有** OpenVINO 最新版。问题出在 QwenPaw 的 pip 版本较旧，或者网络层有缓存。

**解决**：采用多源 fallback 策略：

```bash
# 先升级 pip（确保支持最新镜像协议）
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 配置多源：主源清华 + 备用阿里云 + 官方 PyPI
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set global.extra-index-url "https://mirrors.aliyun.com/pypi/simple/ https://pypi.org/simple/"
```

这样即使某个镜像同步不及时，pip 会自动尝试下一个源，不会卡住。

**教训**：**云环境和本地环境的差异比你想象的大**。本地能跑通不代表云端能跑通。多源 fallback 是对抗环境不确定性的有效手段。

---

## 六、端到端实测

### 6.1 测试环境

| 项 | 值 |
|---|---|
| 操作系统 | macOS (Apple Silicon) / QwenPaw 云端 |
| Python | 3.12 |
| OpenVINO | 2026.2.0 |
| RapidOCR | 1.4.x (ONNX Runtime) |
| Agent 大脑 | Ollama + qwen3.5:9b / QwenPaw 内置 |
| Embedding 模型 | `bge-small-zh-v1.5` INT4，512 维，**19 MB** |

### 6.2 OCR 识别精度

测试素材：用 PIL 渲染的合同扫描件（模拟真实拍照：纸张泛黄 + 旋转 1.3° + 椒盐噪点 + 高斯模糊）

```
$ python tools/ocr.py examples/contract_scan.png
识别行数: 22
平均置信度: 0.9878

[1.00] 智能设备采购与服务合同
[1.00] 合同编号：AIPC-2026-0518
[0.98] 甲方（采购方）：北京智算科技有限公司
[1.00] 5.1乙方逾期交付的，每逾期一日，应向甲方支付合同总价0.5%的违约金，
[1.00] 违约金累计不超过合同总价的10%。
```

22 行全部正确识别，包括 0.5%、10%、12,800 元、256 万元等关键数字。

### 6.3 QwenPaw 云端验证

在 QwenPaw 上直接用自然语言对话测试：

```
用户：帮我解析 examples/contract_scan.pdf，然后告诉我违约金比例是多少

→ 自动检测扫描件（PyMuPDF 返回 0 字符）
→ 调用 RapidOCR 识别全文
→ 建立 FAISS 索引
→ 语义检索违约金条款

回答：
| 违约情形 | 违约金比例 | 上限 |
|---------|----------|------|
| 逾期交付 | 每日合同总价的 0.5% | 累计不超过 10% |
| 拒收/不付款 | 需支付全部款项 + 额外 15% 赔偿金 | - |
| 滞纳金 | 按应付款项计算 | 累计不超过 5% |
```

**QwenPaw 上 Skill 全链路跑通，OCR → 索引 → 检索 → 回答，一气呵成。**

### 6.4 测试用例总览

| # | 场景 | 结果 |
|---|------|------|
| 1 | 纯文本文档问答 (sample.txt) | ✅ |
| 2 | 合同文本问答 (contract_sample.txt) | ✅ |
| 3 | PDF 白皮书问答 (whitepaper_sample.pdf) | ✅ |
| 4 | 不存在的文件 → 正确报错 | ✅ |
| 5 | manifest → OpenAI tools schema 转换 | ✅ |
| 6 | RAG 链路端到端 (BGE + FAISS) | ✅ (无模型时跳过) |
| 7 | 扫描件 PNG OCR 识别 | ✅ |
| 8 | 扫描件 PDF 自动路由到 OCR | ✅ |
| 9 | OCR 引擎 auto fallback | ✅ |
| 10 | QwenPaw 云端端到端对话 | ✅ |

---

## 七、Hybrid AI 的思考：端云协同才是正解

在做这个项目的过程中，我们对"AI 应该跑在哪里"这个问题有了更深的思考。

### 7.1 纯云端的局限

| 问题 | 影响 |
|------|------|
| 数据隐私 | 合同/病历/身份证不能上传 |
| 网络依赖 | 离线场景（飞机、偏远地区）不可用 |
| 成本 | 按页/按 token 收费，大量文档成本高 |
| 延迟 | 网络传输 + 排队等待 |

### 7.2 纯本地的局限

| 问题 | 影响 |
|------|------|
| 模型能力 | 9B 模型的推理能力不如 70B+ |
| 硬件门槛 | 需要 16GB+ 内存的 AI PC |
| 部署复杂 | 用户需要安装 Ollama、下载模型 |
| 维护成本 | 模型更新、依赖升级都需要用户操作 |

### 7.3 Hybrid AI：端云协同

我们最终的架构其实已经体现了 Hybrid AI 的思想：

```
┌─────────────────────────────────────────────┐
│                  云端                        │
│  - QwenPaw 平台：Skill 调度、工作流编排       │
│  - 镜像源：依赖包下载                        │
│  - 模型托管：ModelScope Skills 发布           │
└─────────────────┬───────────────────────────┘
                  │ 网络（仅传输 Skill 定义和指令）
                  ▼
┌─────────────────────────────────────────────┐
│                  本地 AI PC                  │
│  - OCR 引擎：扫描件识别（数据不出机）          │
│  - Embedding：文档向量化（数据不出机）         │
│  - FAISS：语义索引（数据不出机）              │
│  - LLM：9B 小模型回答（数据不出机）           │
└─────────────────────────────────────────────┘
```

**关键洞察**：

> **云端负责"调度"，本地负责"干活"。数据永远不出机，但智能可以来自云端。**

具体来说：
- **Skill 的定义**（SKILL.md）存在云端，由 QwenPaw 读取和调度
- **实际的文档处理**（OCR、Embedding、检索）全部在本地完成
- **用户的文档内容**永远不会上传到任何服务器

这种模式的好处是：
1. **隐私安全**：敏感数据始终在本地
2. **智能上限**：云端大模型负责理解意图和编排工作流，本地小模型负责执行
3. **成本可控**：Skill 调度的 token 消耗极小，主要计算在本地
4. **离线可用**：一旦初始化完成，核心功能可离线运行

### 7.4 对 AI PC 的展望

Intel Core Ultra 处理器内置 NPU（神经网络处理单元），让 AI PC 的本地推理能力大幅提升。我们的 Skill 通过 `device="AUTO"` 已经支持 NPU 自动适配——在 Core Ultra 上，BGE Embedding 和 OCR 推理会自动切换到 NPU，功耗更低、速度更快。

我们相信，未来的 AI 应用一定是 **端云协同** 的：

- **云端**：大模型训练、Skill 生态、跨设备同步
- **本地**：隐私数据处理、实时推理、离线能力
- **NPU**：为 AI 推理专门优化的硬件，让本地 AI 更快、更省电

`scan-and-ask` 只是一个起点。我们期待看到更多"数据不出机"的 AI Skill 出现——合同问答、病历分析、会议纪要、财务审计……每一个涉及敏感文档的场景，都值得用 Hybrid AI 重新做一遍。

---

## 八、踩坑记录

| 坑 | 现象 | 解决 |
|---|---|---|
| PaddleOCR HPI 在 macOS ARM 无包 | `ultra-infer-python` 无 ARM wheel | 替换为 RapidOCR（ONNX Runtime） |
| `chunk_text` 死循环 | `overlap >= chunk_size` 时步长为 0 | 限制 `overlap <= chunk_size // 2` |
| `extract_pdf` 返回值不一致 | 成功 1 值，失败 2 值，解包崩溃 | 统一返回 `(text, error)` 元组 |
| QwenPaw 不调用 Skill 工具 | SKILL.md 写"调 parse_document"，模型不理解 | 改为 shell 命令描述工作流 |
| pip 镜像源找不到 openvino | 单源同步不及时 | 多源 fallback（清华+阿里云+PyPI） |
| PaddlePaddle 安装太大 | 500MB+，云端安装超时 | 替换为 RapidOCR（~30MB） |
| `device="CPU"` 硬编码 | NPU/GPU 不会被使用 | 改为 `device="AUTO"` |

---

## 九、创新点小结

1. **扫描件自动检测 + OCR 路由**：PDF 无文字层时自动切换 OCR，用户无感知
2. **轻量级 OCR（~30MB）**：RapidOCR 替代 PaddlePaddle，安装体积降低 94%，精度不变
3. **硬件自适应（AUTO）**：同一份代码自动适配 CPU/GPU/NPU，零配置
4. **共享 Embedding 模块**：4 个 Tool 共享同一份 BGE 模型，避免重复加载
5. **Anthropic Skills 兼容**：同时支持 QwenPaw 云端和 Ollama 本地两种调用方式
6. **Hybrid AI 架构**：云端调度 + 本地执行，数据不出机，智能不打折

---

## 十、快速体验

```bash
# 一键初始化
bash setup.sh

# 扫描件 OCR 测试
python tools/ocr.py examples/contract_scan.png

# 端到端问答
python run_agent.py "请解析 examples/contract_scan.pdf，违约金怎么算？" \
  --backend ollama --model qwen3.5:9b

# QwenPaw 云端：直接对话即可
# "帮我解析 contract_scan.pdf，违约金比例是多少？"
```

---

## 参考资料

- [OpenVINO 官方文档](https://docs.openvino.ai)
- [RapidOCR GitHub](https://github.com/RapidAI/RapidOCR)
- [Anthropic Agent Skills 规范](https://github.com/anthropics/skills)
- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [QwenPaw 文档](https://qwenpaw.agentscope.io/docs/intro/?lang=zh)
- [Ollama Tool Calling Spec](https://github.com/ollama/ollama/blob/main/docs/api.md)
