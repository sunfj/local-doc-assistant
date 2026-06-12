"""
共享 Embedding 模块
功能：加载 OpenVINO 优化的 BGE 模型 + tokenizer，对文本批量编码为归一化向量
"""
import os
import numpy as np

# 模块级缓存：避免每次调用都重新加载模型
_MODEL_CACHE = {}


def default_model_dir():
    """默认 BGE INT4 OpenVINO 模型目录"""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "models",
        "bge-small-zh-int4",
    )


def load_embedding_model(model_dir=None, device="CPU"):
    """加载 OpenVINO embedding 模型 + tokenizer，带模块级缓存"""
    if model_dir is None:
        model_dir = default_model_dir()
    model_dir = os.path.abspath(model_dir)

    cache_key = (model_dir, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], None

    if not os.path.exists(model_dir):
        return None, (
            f"Embedding 模型不存在: {model_dir}\n"
            "请先运行: optimum-cli export openvino "
            "--model BAAI/bge-small-zh-v1.5 --task feature-extraction "
            f"--weight-format int4 {model_dir}"
        )

    try:
        from openvino import Core
    except Exception as e:
        return None, f"OpenVINO 未安装或导入失败: {e}"

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        return None, f"transformers 未安装: {e}（pip install transformers）"

    try:
        core = Core()
        xml_path = os.path.join(model_dir, "openvino_model.xml")
        model = core.read_model(xml_path)
        compiled_model = core.compile_model(model, device)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        bundle = {
            "compiled_model": compiled_model,
            "tokenizer": tokenizer,
            "input_names": [t.any_name for t in compiled_model.inputs],
            "output_layer": compiled_model.output(0),
            "device": device,
        }
        _MODEL_CACHE[cache_key] = bundle
        return bundle, None
    except Exception as e:
        return None, f"加载 embedding 模型失败: {e}"


def _mean_pooling(last_hidden_state, attention_mask):
    """对最后一层 hidden state 做 mean pooling，忽略 padding"""
    mask = attention_mask.astype(np.float32)[..., None]
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def _l2_normalize(vectors):
    """L2 归一化，BGE 检索约定使用归一化向量做内积/余弦"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def embed_texts(model_bundle, texts, max_length=512, batch_size=8):
    """
    批量编码文本为 (N, dim) 的归一化向量。
    model_bundle: load_embedding_model 返回的字典
    texts: List[str]
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    compiled = model_bundle["compiled_model"]
    tokenizer = model_bundle["tokenizer"]
    input_names = set(model_bundle["input_names"])

    all_vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        feed = {}
        if "input_ids" in input_names:
            feed["input_ids"] = enc["input_ids"].astype(np.int64)
        if "attention_mask" in input_names:
            feed["attention_mask"] = enc["attention_mask"].astype(np.int64)
        if "token_type_ids" in input_names:
            tti = enc.get("token_type_ids")
            if tti is None:
                tti = np.zeros_like(enc["input_ids"])
            feed["token_type_ids"] = tti.astype(np.int64)

        outputs = compiled(feed)
        # 取第一个输出（last_hidden_state），形状 [B, T, H]
        last_hidden = list(outputs.values())[0]
        pooled = _mean_pooling(last_hidden, enc["attention_mask"])
        all_vectors.append(pooled.astype(np.float32))

    vectors = np.concatenate(all_vectors, axis=0)
    return _l2_normalize(vectors)


def embed_text(model_bundle, text, max_length=512):
    """单条文本编码，返回 (dim,) 一维向量"""
    vec = embed_texts(model_bundle, [text], max_length=max_length, batch_size=1)
    return vec[0]


def embedding_dim(model_bundle):
    """获取向量维度（通过试编码一次推断）"""
    vec = embed_text(model_bundle, "dim_probe")
    return int(vec.shape[0])
