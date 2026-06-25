# 小红书发布文案

## 标题（二选一）

**方案A**：合同照片一拍，AI 帮你查违约金🔍 本地 OCR+RAG 文档问答 Skill 实战

**方案B**：数据不出机！我做了一个"拍照即问"的本地 AI 文档助手 📄

---

## 正文

折腾了一个 Skill，效果超出预期 ✨

**一句话说清楚**：拍张合同照片 / 丢个扫描件 PDF，AI 直接告诉你"违约金是多少"、"甲方是谁"——全程本地处理，数据不出电脑 🔒

### 🛠️ 技术栈

```
OCR：RapidOCR（PP-OCRv4，仅 30MB）
Embedding：BGE-small-zh INT4（19MB）
向量库：FAISS
推理：OpenVINO（自动适配 CPU/GPU/NPU）
Agent：Qwen3.5-9B
```

### 📐 工作流

```
扫描件/照片 → OCR 识别文字 → 向量建索引 → 语义检索 → AI 回答
     ↑                                              ↑
  自动检测                                      数据不出机
```

### ✅ 实测效果

丢了一份扫描件合同 PDF（无文字层），问"违约金比例是多少"：

> AI 回答：
> 
> - 逾期交付：每日 0.5%，累计不超过 10%
> - 拒收/不付款：额外赔偿 15%
> - 滞纳金：累计不超过 5%

OCR 识别 22 行，置信度 98.78%，关键数字全部正确 🎯

### 💡 三个优化心得

1️⃣ **OCR 选型**：PaddleOCR 装 500MB，RapidOCR 只要 30MB，精度一样（同一模型）
2️⃣ **硬件自适配**：`device="AUTO"` 一行代码，Mac/Intel NPU/Arc GPU 自动切换
3️⃣ **多源镜像**：云端环境 pip 装包总翻车？清华+阿里云+PyPI 三源 fallback 稳了

### 🤔 Hybrid AI 的思考

> 云端负责调度，本地负责干活。数据永远不出机，但智能可以来自云端。

隐私敏感场景（合同/病历/身份证），AI 不一定非要"上云"。端云协同才是正解。

---

### 🔗 链接

📦 Skill：https://modelscope.cn/skills/seunal/scan-and-ask
📖 技术文章：github.com/sunfj/local-doc-assistant
🏷️ 项目：scan-and-ask（扫描即问，本地 AI 文档问答）

---

## 话题标签

#英特尔 #openvino #魔搭社区 #modelscope #agentic #skills #AI PC #本地AI #文档问答 #OCR #隐私保护

## @提及

@OpenVINO中文社区 @魔搭ModelScope社区

---

## 配图建议（共 4-6 张）

| 序号  | 内容           | 说明                         |
| --- | ------------ | -------------------------- |
| 图1  | 项目封面/标题图     | "scan-and-ask 扫描即问" + 架构简图 |
| 图2  | 工作流程图        | 4 步流程：OCR → 索引 → 检索 → 回答   |
| 图3  | QwenPaw 测试截图 | 对话界面 + AI 回答（你提供）          |
| 图4  | OCR 识别结果截图   | 终端输出 22 行识别结果              |
| 图5  | 技术栈对比表       | PaddleOCR vs RapidOCR 体积对比 |
| 图6  | 10 个测试用例全部通过 | pytest 输出截图                |
