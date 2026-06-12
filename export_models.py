"""
模型导出脚本：下载 HuggingFace 原始模型 → 转换为 OpenVINO INT4
================================================================

用法:
  python export_models.py                                     # 导出两个模型
  python export_models.py --model bge                          # 仅导出 embedding
  python export_models.py --model qwen                         # 仅导出 LLM
  python export_models.py --model bge --device CPU             # 指定导出设备

注意：
  - Qwen2.5-7B-Instruct INT4 导出约需 8GB RAM + 8GB 磁盘空间
  - 8GB RAM 的老机器可能 OOM，建议在 ≥16GB 的机器上导出后复制 models/ 目录
  - 导出耗时：bge-small 约 2-5 分钟，Qwen2.5-7B 约 10-30 分钟（视网速和 CPU）
  - 导出后模型置于 local-doc-assistant/models/ 下，运行时自动加载
"""
import os
import sys
import argparse
import subprocess


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


BGE_CONFIG = {
    "name": "BAAI/bge-small-zh-v1.5",
    "task": "feature-extraction",
    "out_dir": os.path.join(MODELS_DIR, "bge-small-zh-int4"),
    "quantize": True,
}

QWEN_CONFIG = {
    "name": "Qwen/Qwen2.5-7B-Instruct",
    "task": "text-generation-with-past",
    "out_dir": os.path.join(MODELS_DIR, "qwen2.5-7b-int4"),
    "quantize": True,
    "extra_args": ["--group-size", "64", "--ratio", "1.0"],
}


def _find_optimum_cli():
    """在当前 Python 同级 bin 目录或系统 PATH 中查找 optimum-cli"""
    import shutil

    # 1) 当前 Python 解释器同级目录
    py_bin = os.path.dirname(sys.executable)
    candidate = os.path.join(py_bin, "optimum-cli")
    if os.path.exists(candidate):
        return candidate
    # 2) 系统 PATH
    return shutil.which("optimum-cli")


def run_optimum_export(cfg):
    """调用 optimum-cli export openvino 转换单个模型"""
    out_dir = cfg["out_dir"]
    if os.path.exists(os.path.join(out_dir, "openvino_model.xml")):
        print(f"  ✓ 模型已存在，跳过: {cfg['name']} → {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    # 优先用 optimum-cli 可执行文件；若不在 PATH 则 fallback 到 python -m optimum.commands
    optimum_cli = _find_optimum_cli()
    if optimum_cli:
        cmd = [optimum_cli, "export", "openvino"]
    else:
        cmd = [sys.executable, "-m", "optimum.commands", "export", "openvino"]
    cmd += [
        "--model", cfg["name"],
        "--task", cfg["task"],
        out_dir,
    ]
    if cfg.get("quantize"):
        cmd += ["--weight-format", "int4"]
    for arg in cfg.get("extra_args", []):
        cmd.append(arg)

    print(f"  → 导出 {cfg['name']} ...")
    print(f"    命令行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"导出失败 (exit={result.returncode}): {cfg['name']}")

    print(f"  ✓ 导出完成: {cfg['name']} → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="导出 OpenVINO INT4 模型")
    parser.add_argument(
        "--model",
        choices=["bge", "qwen", "all"],
        default="all",
        help="要导出的模型：bge / qwen / all（默认 all）",
    )
    args = parser.parse_args()

    models_to_export = []
    if args.model in ("bge", "all"):
        models_to_export.append(BGE_CONFIG)
    if args.model in ("qwen", "all"):
        models_to_export.append(QWEN_CONFIG)

    if not models_to_export:
        print("未选择要导出的模型。")
        return

    print("=" * 60)
    print("  OpenVINO INT4 模型导出工具")
    print("  请确保 pip install optimum[openvino,nncf]")
    print("=" * 60)

    for cfg in models_to_export:
        run_optimum_export(cfg)
        print()

    print("所有模型导出完成。目录结构：")
    for cfg in models_to_export:
        out = cfg["out_dir"]
        size_mb = 0
        if os.path.exists(out):
            for root, dirs, files in os.walk(out):
                for f in files:
                    fp = os.path.join(root, f)
                    size_mb += os.path.getsize(fp)
            print(f"  {os.path.relpath(out, PROJECT_ROOT)}  ({size_mb / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()