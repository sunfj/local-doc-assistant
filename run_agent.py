"""
端到端 Agent 示例：Ollama (Qwen3.6-35B-A3B) + local-doc-assistant Skill

流程：
  1. 加载 manifest.json，转换为 OpenAI 兼容的 tools schema
  2. 用户用自然语言提问（例如：请总结 docs/sample.txt 的核心观点）
  3. Qwen3.6 决定调用 parse_document → build_index → query_document
  4. 本脚本捕获每次 tool_calls，分发到 tools/*.py 的 main(args)
  5. 把工具结果以 role=tool 回传给模型，直到模型给出最终回答

依赖：本机已运行 Ollama 服务，并已 pull qwen3:30b-a3b-instruct（或类似 Qwen3.6 模型）
"""
import os
import sys
import json
import importlib.util
import argparse
import requests


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)


def load_manifest(manifest_path):
    """读取 manifest.json"""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def manifest_to_openai_tools(manifest):
    """把 Skill manifest.tools 转换为 OpenAI/Ollama tools schema"""
    tools = []
    for t in manifest["tools"]:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
        )
    return tools


def load_tool_entry(entry_point):
    """根据 'tools/xxx.py:main' 动态加载 main 函数"""
    rel_path, func_name = entry_point.split(":")
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(rel_path, abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def build_tool_registry(manifest):
    """name -> callable(args_dict) 的注册表"""
    registry = {}
    for t in manifest["tools"]:
        registry[t["name"]] = load_tool_entry(t["entry_point"])
    return registry


def call_ollama(base_url, model, messages, tools):
    """调用 Ollama /api/chat 接口，启用 tool calling"""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def run_agent(
    user_query,
    manifest_path,
    ollama_url="http://localhost:11434",
    model="qwen3:30b-a3b-instruct",
    max_steps=8,
):
    manifest = load_manifest(manifest_path)
    tools_schema = manifest_to_openai_tools(manifest)
    registry = build_tool_registry(manifest)

    # 读取 SKILL.md 作为 system prompt
    skill_md = os.path.join(os.path.dirname(manifest_path), "SKILL.md")
    system_prompt = ""
    if os.path.exists(skill_md):
        with open(skill_md, "r", encoding="utf-8") as f:
            system_prompt = f.read()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n=== Step {step}: 调用 Qwen 决策 ===")
        response = call_ollama(ollama_url, model, messages, tools_schema)
        msg = response.get("message", {})
        tool_calls = msg.get("tool_calls") or []

        # 追加助手消息（含可能的 tool_calls）到对话
        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content", ""),
                "tool_calls": tool_calls,
            }
        )

        if not tool_calls:
            print("\n>>> 最终回答：")
            print(msg.get("content", ""))
            return msg.get("content", "")

        # 顺序执行所有 tool_calls
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}

            print(f"  → 调用工具: {name}({args})")
            if name not in registry:
                tool_result = json.dumps(
                    {"status": "error", "message": f"未知工具: {name}"},
                    ensure_ascii=False,
                )
            else:
                try:
                    tool_result = registry[name](args)
                except Exception as e:
                    tool_result = json.dumps(
                        {"status": "error", "message": f"工具执行异常: {e}"},
                        ensure_ascii=False,
                    )

            print(f"  ← 结果: {tool_result[:200]}{'...' if len(tool_result) > 200 else ''}")
            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": tool_result,
                }
            )

    print("\n⚠️ 达到最大步数仍未给出最终回答")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="local-doc-assistant 端到端 Agent 示例")
    parser.add_argument(
        "query",
        nargs="?",
        default="请解析 examples/sample.txt 并告诉我 OpenVINO 支持哪些硬件设备？",
        help="用户提问",
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join(PROJECT_ROOT, "manifest.json"),
        help="Skill manifest.json 路径",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AGENT_MODEL", "qwen3:30b-a3b-instruct"),
        help="Ollama 模型名（≤35B），如 qwen3:30b-a3b-instruct / qwen2.5:7b-instruct",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    run_agent(
        user_query=args.query,
        manifest_path=args.manifest,
        ollama_url=args.ollama_url,
        model=args.model,
        max_steps=args.max_steps,
    )
