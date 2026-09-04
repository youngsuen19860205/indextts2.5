# IndexTTS 2.5 批量语音克隆运行环境（NVIDIA H20 / Hopper, sm_90）
#
# 说明：镜像内自带 CUDA 运行库，只要求宿主机安装了兼容的 NVIDIA 驱动
# （CUDA 12.8 需要 >= 525 系列驱动，H20 推荐 535+）和 NVIDIA Container Toolkit，
# 宿主机**不需要**安装相同版本的 CUDA Toolkit。
#
# 模型权重不打进镜像，运行时通过挂载 ./checkpoints 提供。
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}" \
    PYTHONPATH=/workspace \
    HF_HOME=/cache/huggingface \
    MODELSCOPE_CACHE=/cache/modelscope \
    INDEXTTS_MODEL_DIR=/workspace/checkpoints

# 系统依赖：音频处理、构建工具、git(-lfs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libsndfile1 \
        pkg-config \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# uv 是上游官方唯一支持的依赖管理方式
RUN curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh

WORKDIR /workspace

# 先同步依赖（利用 Docker 层缓存），依赖版本由上游 uv.lock 锁定（torch cu128）
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# 再拷贝项目代码
COPY . /workspace
RUN uv sync --frozen --no-dev

RUN mkdir -p /workspace/checkpoints /workspace/reference /workspace/gen_text /workspace/gen_wav \
             /cache/huggingface /cache/modelscope

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["check-gpu"]
