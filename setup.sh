#!/usr/bin/env bash
# local-doc-assistant Skill 一键初始化脚本
# 作用：安装依赖 + 从 ModelScope 下载 BGE 模型 + 转换为 OpenVINO INT4
#
# 用法：
#   bash setup.sh                 # 全套初始化
#   bash setup.sh --check         # 仅检查环境，不安装
#
# 设计目标：在 QwenPaw 云主机首次激活 Skill 时，由模型用 execute_shell_command
# 调用一次本脚本即可完成全部环境准备。

set -e

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SKILL_DIR}/models"
BGE_DIR="${MODELS_DIR}/bge-small-zh-int4"
PY="${PY:-python3}"

print_header() {
  echo ""
  echo "=================================================="
  echo "  $1"
  echo "=================================================="
}

check_env() {
  print_header "检查运行环境"
  echo "Python:   $(${PY} --version 2>&1)"
  echo "Skill 目录: ${SKILL_DIR}"

  echo ""
  echo "依赖检查："
  for pkg in openvino faiss transformers fitz numpy rapidocr_onnxruntime; do
    if ${PY} -c "import ${pkg}" 2>/dev/null; then
      echo "  ✓ ${pkg}"
    else
      echo "  ✗ ${pkg} (未安装)"
    fi
  done

  echo ""
  if [ -f "${BGE_DIR}/openvino_model.xml" ]; then
    echo "BGE 模型：✓ 已就绪 (${BGE_DIR})"
  else
    echo "BGE 模型：✗ 未导出 (${BGE_DIR})"
  fi
}

install_deps() {
  print_header "配置 pip 国内镜像源（加速下载）"
  ${PY} -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
  ${PY} -m pip config set global.trusted-host mirrors.aliyun.com

  print_header "安装 Python 依赖"
  ${PY} -m pip install --upgrade pip
  ${PY} -m pip install -r "${SKILL_DIR}/requirements.txt"
  # 模型转换需要 optimum[openvino]
  ${PY} -m pip install "optimum[openvino,nncf]" modelscope
  # RapidOCR（轻量级 OCR，基于 PaddleOCR ONNX 模型，无需 PaddlePaddle）
  ${PY} -m pip install rapidocr-onnxruntime
}

download_bge_modelscope() {
  print_header "从 ModelScope 下载 BGE-small-zh-v1.5"
  local DL_DIR="${MODELS_DIR}/_bge_hf_src"
  mkdir -p "${MODELS_DIR}"

  ${PY} - <<EOF
from modelscope import snapshot_download
import os
target = "${DL_DIR}"
os.makedirs(target, exist_ok=True)
path = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir=target)
print(f"DOWNLOADED_TO={path}")
EOF
}

convert_to_openvino_int4() {
  print_header "转换为 OpenVINO INT4 格式"
  # 找到刚下载的 HF 模型本地路径（snapshot_download 的目录结构是 cache_dir/<owner>/<name>）
  local HF_PATH
  HF_PATH=$(find "${MODELS_DIR}/_bge_hf_src" -maxdepth 4 -name "config.json" -path "*bge-small-zh*" ! -path "*/1_Pooling/*" | head -1)
  HF_PATH=$(dirname "${HF_PATH}")

  if [ -z "${HF_PATH}" ] || [ ! -f "${HF_PATH}/config.json" ]; then
    echo "未找到下载的 HF 模型；尝试用 optimum 直接拉。"
    HF_PATH="BAAI/bge-small-zh-v1.5"
  fi

  echo "源模型：${HF_PATH}"
  echo "输出： ${BGE_DIR}"

  # 查找 optimum-cli 可执行文件
  local OPT_CLI
  OPT_CLI="$(dirname "${PY}")/optimum-cli"
  if [ ! -x "${OPT_CLI}" ]; then
    OPT_CLI="$(command -v optimum-cli 2>/dev/null || echo "")"
  fi

  if [ -n "${OPT_CLI}" ] && [ -x "${OPT_CLI}" ]; then
    echo "使用 optimum-cli: ${OPT_CLI}"
    "${OPT_CLI}" export openvino \
      --model "${HF_PATH}" \
      --task feature-extraction \
      --weight-format int4 \
      "${BGE_DIR}"
  else
    echo "optimum-cli 不可用，跳过模型转换"
    echo "请手动运行: pip install optimum[openvino,nncf] && optimum-cli export openvino --model ${HF_PATH} --task feature-extraction --weight-format int4 ${BGE_DIR}"
    return 1
  fi

  # 清理下载缓存（节省磁盘）
  rm -rf "${MODELS_DIR}/_bge_hf_src"
}

verify() {
  print_header "验证 Skill 链路"
  ${PY} "${SKILL_DIR}/examples/smoke_test.py"
  echo ""
  print_header "验证 OCR 链路"
  ${PY} "${SKILL_DIR}/examples/generate_scan_samples.py"
  ${PY} "${SKILL_DIR}/tools/ocr.py" "${SKILL_DIR}/examples/contract_scan.png" 2>&1 | head -5
}

main() {
  case "${1:-}" in
    --check)
      check_env
      ;;
    *)
      install_deps
      if [ ! -f "${BGE_DIR}/openvino_model.xml" ]; then
        download_bge_modelscope || true
        convert_to_openvino_int4
      else
        echo "BGE 模型已存在，跳过下载"
      fi
      verify
      ;;
  esac

  print_header "完成"
  echo "现在可以使用以下命令测试：(在 ${SKILL_DIR} 目录下)"
  echo "  python tools/doc_parser.py examples/sample.txt"
  echo "  python tools/vector_store.py <doc_id>"
  echo "  python tools/rag_query.py <doc_id> 'OpenVINO 支持哪些设备?'"
}

main "$@"
