"""
本地 OpenVINO LLM 推理模块
功能：使用 OpenVINO GenAI LLMPipeline 在本机运行 Qwen2.5-7B INT4 等生成模型
"""
import os
import json

_PIPELINE_CACHE = {}


def default_llm_model_dir():
    """默认 Qwen2.5-7B INT4 OpenVINO 模型目录"""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "models",
        "qwen2.5-7b-int4",
    )


def load_llm_pipeline(model_dir=None, device="AUTO"):
    """加载 OpenVINO GenAI LLMPipeline，带模块级缓存
    device: AUTO（默认，自动选最快硬件）/ CPU / GPU / NPU
    """
    if model_dir is None:
        model_dir = default_llm_model_dir()
    model_dir = os.path.abspath(model_dir)

    cache_key = (model_dir, device)
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key], None

    if not os.path.exists(model_dir):
        return None, (
            f"LLM 模型不存在: {model_dir}\n"
            "请先运行: python export_models.py --model qwen"
        )

    try:
        import openvino_genai as ov_genai
    except Exception as e:
        return None, f"openvino-genai 未安装或导入失败: {e}"

    try:
        pipe = ov_genai.LLMPipeline(model_dir, device)
        _PIPELINE_CACHE[cache_key] = pipe
        return pipe, None
    except Exception as e:
        return None, f"加载 OpenVINO LLM 失败: {e}"


def _messages_to_prompt(messages):
    """将 OpenAI messages 简单转换为 Qwen/通用 chat prompt"""
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lines.append(f"系统：{content}")
        elif role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            lines.append(f"助手：{content}")
        elif role == "tool":
            name = msg.get("name", "tool")
            lines.append(f"工具结果（{name}）：{content}")
    lines.append("助手：")
    return "\n\n".join(lines)


def generate(messages, model_dir=None, device="AUTO", max_new_tokens=512, temperature=0.2):
    """生成最终回答。注意：当前 OpenVINO backend 只负责最终汇总，不做 tool calling 决策。"""
    pipe, error = load_llm_pipeline(model_dir=model_dir, device=device)
    if error:
        return None, error

    prompt = _messages_to_prompt(messages)
    try:
        result = pipe.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        return str(result), None
    except Exception as e:
        return None, f"OpenVINO LLM 生成失败: {e}"


def main(args):
    """Tool 入口：供外部直接调用本地 LLM 汇总上下文"""
    messages = args.get("messages", [])
    if not messages:
        prompt = args.get("prompt", "")
        messages = [{"role": "user", "content": prompt}]
    text, error = generate(
        messages,
        model_dir=args.get("model_dir"),
        device=args.get("device", "AUTO"),
        max_new_tokens=args.get("max_new_tokens", 512),
        temperature=args.get("temperature", 0.2),
    )
    if error:
        return json.dumps({"status": "error", "message": error}, ensure_ascii=False)
    return json.dumps({"status": "success", "text": text}, ensure_ascii=False)
