#!/usr/bin/env bash
# 下载 IndexTTS 2.5 权重（IndexTeam/IndexTTS-2.5）。
#
# 用法:
#   scripts/download_models.sh [huggingface|modelscope] [模型目录]
#
# 环境变量:
#   INDEXTTS_MODEL_DIR  默认模型目录（默认 ./checkpoints）
#   HF_ENDPOINT         Hugging Face 镜像，例如 https://hf-mirror.com
set -Eeuo pipefail

SOURCE="${1:-${INDEXTTS_DOWNLOAD_SOURCE:-huggingface}}"
MODEL_DIR="${2:-${INDEXTTS_MODEL_DIR:-./checkpoints}}"
REPO_ID="IndexTeam/IndexTTS-2.5"

mkdir -p "$MODEL_DIR"

run_python() {
    if command -v uv >/dev/null 2>&1 && [ -f "pyproject.toml" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
        uv run python "$@"
    else
        python "$@"
    fi
}

case "$SOURCE" in
    huggingface | hf)
        echo ">> 从 Hugging Face 下载 ${REPO_ID} 到 ${MODEL_DIR}"
        run_python - "$REPO_ID" "$MODEL_DIR" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
print(f">> 下载完成: {path}")
PY
        ;;
    modelscope | ms)
        echo ">> 从 ModelScope 下载 ${REPO_ID} 到 ${MODEL_DIR}"
        run_python - "$REPO_ID" "$MODEL_DIR" <<'PY'
import sys

from modelscope.hub.snapshot_download import snapshot_download

model_id, local_dir = sys.argv[1], sys.argv[2]
path = snapshot_download(model_id, local_dir=local_dir)
print(f">> 下载完成: {path}")
PY
        ;;
    *)
        echo "未知下载源: ${SOURCE}（支持 huggingface | modelscope）" >&2
        exit 2
        ;;
esac

missing=()
for file in gpt.pth s2mel.pth codec.pth multilingual_zh_ja_yue_char_del.tiktoken wav2vec2bert_stats.pt config.yaml; do
    [ -f "${MODEL_DIR}/${file}" ] || missing+=("$file")
done

if [ "${#missing[@]}" -gt 0 ]; then
    echo "!! 模型目录 ${MODEL_DIR} 仍缺少: ${missing[*]}" >&2
    exit 1
fi

echo ">> IndexTTS 2.5 权重就绪: ${MODEL_DIR}"
