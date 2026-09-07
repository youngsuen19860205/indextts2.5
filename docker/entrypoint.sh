#!/usr/bin/env bash
# 容器入口：默认打印 CUDA/PyTorch/GPU 状态，也可直接执行批量推理或任意命令。
set -Eeuo pipefail

MODEL_DIR="${INDEXTTS_MODEL_DIR:-/workspace/checkpoints}"

usage() {
    cat <<'EOF'
用法: docker run ... indextts2.5 <命令> [参数...]

命令:
  check-gpu            打印 CUDA / PyTorch / GPU 可用性（默认）
  download [source]    下载 IndexTTS 2.5 权重，source = huggingface|modelscope
  dry-run [参数...]    校验 reference/gen_text 配对并打印任务计划（不加载模型）
  generate [参数...]   执行批量语音克隆
  random-clone [参数...] 从 speaker_wav_txt/gen_txt 随机选择并生成克隆语音
  shell                进入交互式 bash
  <其他>               直接作为命令执行
EOF
}

check_gpu() {
    echo ">> nvidia-smi:"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi || echo "!! nvidia-smi 执行失败，请确认使用了 --gpus all"
    else
        echo "!! 容器内未找到 nvidia-smi，请确认已安装 NVIDIA Container Toolkit"
    fi
    python - <<'PY'
import torch

print(">> torch:", torch.__version__)
print(">> torch CUDA 版本:", torch.version.cuda)
print(">> CUDA 可用:", torch.cuda.is_available())
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f">> GPU {index}: {props.name} ({props.total_memory / 1024 ** 3:.1f} GB, sm_{props.major}{props.minor})")
PY
}

command="${1:-check-gpu}"
[ "$#" -gt 0 ] && shift || true

case "$command" in
    check-gpu)
        check_gpu
        ;;
    download)
        exec /workspace/scripts/download_models.sh "${1:-huggingface}" "$MODEL_DIR"
        ;;
    dry-run)
        exec python -m indextts_batch --model-dir "$MODEL_DIR" --dry-run "$@"
        ;;
    generate)
        exec python -m indextts_batch --model-dir "$MODEL_DIR" "$@"
        ;;
    random-clone)
        exec python -m indextts_batch.random_clone --model-dir "$MODEL_DIR" "$@"
        ;;
    shell)
        exec /bin/bash
        ;;
    -h | --help | help)
        usage
        ;;
    *)
        exec "$command" "$@"
        ;;
esac
