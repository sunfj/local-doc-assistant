"""
端到端 Agent 示例：local-doc-assistant Skill

支持两种后端：
  1. ollama   ：Qwen3.6/Qwen2.5 通过 Tool Calling 自动规划并调用工具
  2. openvino ：手动执行 parse/build/query，最后用 OpenVINO GenAI LLMPipeline 本地汇总

推荐比赛验证：--backend ollama（贴近 Qwen3.6-35B-A3B Tool Calling）
推荐全本地演示：--backend openvino（需先导出 Qwen2.5-7B INT4）
"""
import os
import sys
import json
import importlib.util
import argparse
import re
import requests


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)


EXAMPLE_FILE = os.path.join(PROJECT_ROOT, "examples", "sample.txt")


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
    payload = {"model": model, "messages": messages, "tools": tools, "stream": False}
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def run_ollama_agent(
    user_query,
    manifest_path,
    ollama_url="http://localhost:11434",
    model="qwen3:30b-a3b-instruct",
    max_steps=8,
):
    """Ollama/Qwen 后端：让 LLM 自动 tool calling"""
    manifest = load_manifest(manifest_path)
    tools_schema = manifest_to_openai_tools(manifest)
    registry = build_tool_registry(manifest)

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
        print(f"\n=== Step {step}: Ollama/Qwen 决策 ===")
        response = call_ollama(ollama_url, model, messages, tools_schema)
        msg = response.get("message", {})
        tool_calls = msg.get("tool_calls") or []

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
                    {"status": "error", "message": f"未知工具: {name}"}, ensure_ascii=False
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
            messages.append({"role": "tool", "name": name, "content": tool_result})

    print("\n⚠️ 达到最大步数仍未给出最终回答")
    return None


def _extract_file_path(query):
    """从用户问题中提取文件路径；没有路径时使用 examples/sample.txt"""
    patterns = [r"([^\s，。；,;]+\.(?:pdf|txt|md|markdown))", r"['\"]([^'\"]+)['\"]"]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            path = match.group(1)
            if not os.path.isabs(path):
                path = os.path.join(PROJECT_ROOT, path)
            return path
    return EXAMPLE_FILE


def run_openvino_agent(user_query, manifest_path, device="CPU", llm_model_dir=None, top_k=3):
    """
    OpenVINO 后端：不用外部 Agent，手动完成 RAG，再用 OpenVINO LLM 汇总。
    适合展示「纯本地」闭环；需要先导出 qwen2.5-7b-int4。
    """
    manifest = load_manifest(manifest_path)
    registry = build_tool_registry(manifest)

    file_path = _extract_file_path(user_query)
    print(f"\n=== Step 1: parse_document({file_path}) ===")
    parse_result = json.loads(registry["parse_document"]({"file_path": file_path}))
    print(parse_result)
    if parse_result.get("status") != "success":
        return parse_result

    doc_id = parse_result["doc_id"]
    print(f"\n=== Step 2: build_index({doc_id}) ===")
    build_result = json.loads(registry["build_index"]({"doc_id": doc_id, "device": device}))
    print(build_result)
    if build_result.get("status") != "success":
        return build_result

    print(f"\n=== Step 3: query_document({doc_id}) ===")
    query_result = json.loads(
        registry["query_document"](
            {"doc_id": doc_id, "query": user_query, "top_k": top_k, "device": device}
        )
    )
    print(json.dumps(query_result, ensure_ascii=False, indent=2)[:1200])
    if query_result.get("status") != "success":
        return query_result

    context = "\n\n".join(
        f"[片段 {item['rank']}, score={item['score']:.4f}]\n{item['text']}"
        for item in query_result.get("results", [])
    )
    messages = [
        {
            "role": "system",
            "content": "你是本地文档分析助手。只能基于给定检索片段回答，不要编造。",
        },
        {
            "role": "user",
            "content": f"用户问题：{user_query}\n\n检索片段：\n{context}\n\n请用中文给出准确、简洁的回答。",
        },
    ]

    print("\n=== Step 4: OpenVINO LLM 汇总 ===")
    from llm import generate

    answer, error = generate(messages, model_dir=llm_model_dir, device=device)
    if error:
        print(f"OpenVINO LLM 暂不可用：{error}")
        print("\n>>> 检索结果已完成，以下为基于片段的简要回答：")
        fallback = context[:1200]
        print(fallback)
        return {"status": "partial", "message": error, "retrieval": query_result}

    print("\n>>> 最终回答：")
    print(answer)
    return {"status": "success", "answer": answer, "retrieval": query_result}


def run_agent(user_query, manifest_path, backend="ollama", **kwargs):
    """统一入口"""
    if backend == "ollama":
        return run_ollama_agent(user_query, manifest_path, **kwargs)
    if backend == "openvino":
        return run_openvino_agent(user_query, manifest_path, **kwargs)
    raise ValueError(f"未知 backend: {backend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="local-doc-assistant 端到端 Agent 示例")
    parser.add_argument(
        "query",
        nargs="?",
        default="请解析 examples/sample.txt 并告诉我 OpenVINO 支持哪些硬件设备？",
        help="用户提问",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "openvino"],
        default="ollama",
        help="后端：ollama 自动 tool calling；openvino 手动 RAG + 本地 LLM 汇总",
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join(PROJECT_ROOT, "manifest.json"),
        help="Skill manifest.json 路径",
    )
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument(
        "--model",
        default=os.environ.get("AGENT_MODEL", "qwen3:30b-a3b-instruct"),
        help="Ollama 模型名（≤35B）",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--device", default="CPU", help="OpenVINO 设备 CPU/GPU/NPU")
    parser.add_argument("--llm-model-dir", default=None, help="OpenVINO LLM 模型目录")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.backend == "ollama":
        run_ollama_agent(
            user_query=args.query,
            manifest_path=args.manifest,
            ollama_url=args.ollama_url,
            model=args.model,
            max_steps=args.max_steps,
        )
    else:
        run_openvino_agent(
            user_query=args.query,
            manifest_path=args.manifest,
            device=args.device,
            llm_model_dir=args.llm_model_dir,
            top_k=args.top_k,
        )
