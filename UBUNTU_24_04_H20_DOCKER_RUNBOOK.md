# Ubuntu 24.04 + NVIDIA H20 + CUDA 13.0：Docker 部署与 WAV 生成教程

本文档面向以下宿主机环境，按步骤完成 Docker 环境创建、GPU 容器配置、IndexTTS 2.5 依赖安装、模型下载、数据准备和 WAV 生成：

```text
操作系统：Ubuntu 24.04.3 LTS
宿主机驱动报告的 CUDA 上限：CUDA 13.0
GPU PCI 设备：NVIDIA Corporation GH100 [H20] (rev a1)
项目分支：copilot/create-index-tts-25-project
```

> 本文档编写于 2026-09-07。命令以服务器上的普通 sudo 用户执行。

## 1. 先理解宿主机 CUDA 13.0 与容器 CUDA 12.8

本项目的 `docker/Dockerfile` 使用：

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
```

这与宿主机显示 `CUDA 13.0` **不冲突**：

- `nvidia-smi` 显示的 `CUDA Version: 13.0` 通常代表当前 NVIDIA 驱动支持的最高 CUDA 版本，不等于项目必须使用 CUDA 13.0 编译或运行；
- NVIDIA 驱动具有向后兼容能力，支持 CUDA 13.0 的新驱动可以运行使用 CUDA 12.8 runtime 的容器应用；
- 宿主机是 Ubuntu 24.04.3，容器内部使用 Ubuntu 22.04 是正常的，容器拥有自己的用户态系统库；
- GPU 驱动由宿主机提供，CUDA runtime、cuDNN、Python 和 PyTorch 依赖放在容器中；
- 不建议为了“版本数字一致”而把当前 Dockerfile 强行改成 CUDA 13.0，因为本项目上游 `uv.lock` 当前锁定的是 PyTorch CUDA 12.8（cu128）依赖。

因此推荐组合为：

```text
宿主机：Ubuntu 24.04.3 + 可报告 CUDA 13.0 的 NVIDIA 驱动
容器：Ubuntu 22.04 + CUDA 12.8.1 + cuDNN + PyTorch cu128
GPU：NVIDIA H20 / Hopper
```

官方参考：

- NVIDIA CUDA compatibility：<https://docs.nvidia.com/deploy/cuda-compatibility/latest/>
- NVIDIA driver / CUDA matrix：<https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html>
- NVIDIA Container Toolkit：<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
- Docker Engine on Ubuntu：<https://docs.docker.com/engine/install/ubuntu/>

## 2. 检查服务器基础信息

执行：

```bash
cat /etc/os-release
uname -a
lspci | grep -i nvidia
nvidia-smi
```

预期至少满足：

1. 系统为 Ubuntu 24.04；
2. `lspci` 能看到 `NVIDIA Corporation GH100 [H20]`；
3. `nvidia-smi` 可以正常运行；
4. `nvidia-smi` 能显示 GPU、驱动版本和 CUDA 13.0；
5. 没有 `NVIDIA-SMI has failed`、驱动未加载或设备不可见错误。

记录驱动信息：

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

如果 `nvidia-smi` 失败，请先修复宿主机 NVIDIA 驱动，再继续安装 Docker。不要在 Docker 镜像中安装 Linux 内核驱动。

## 3. 安装 Docker Engine

如果服务器已经正确安装 Docker Engine 和 Compose，可跳到第 4 步。

### 3.1 删除可能冲突的软件包

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  sudo apt-get remove -y "$pkg" || true
done
```

### 3.2 添加 Docker 官方 apt 仓库

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

. /etc/os-release
printf '%s\n' \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
```

### 3.3 安装 Docker 和 Compose 插件

```bash
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin
```

启动并检查：

```bash
sudo systemctl enable --now docker
sudo systemctl status docker --no-pager
docker --version
sudo docker compose version
sudo docker run --rm hello-world
```

### 3.4 可选：允许当前用户不使用 sudo 执行 Docker

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

重新登录服务器后确认：

```bash
docker run --rm hello-world
docker compose version
```

> `docker` 用户组基本等同于 root 权限。生产服务器应根据安全策略决定是否执行此步骤；不加入时，将后续命令中的 `docker` 改为 `sudo docker`。

## 4. 安装 NVIDIA Container Toolkit

Docker 默认不能把 H20 映射到容器，需要安装 NVIDIA Container Toolkit。

### 4.1 添加 NVIDIA 软件源

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 4.2 配置 Docker GPU runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

检查 Docker 配置：

```bash
docker info | grep -i runtime -A 3
```

### 4.3 在容器中验证 H20

使用与本项目一致的 CUDA 12.8 容器验证：

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu22.04 \
  nvidia-smi
```

预期容器内同样能看到 H20。即使宿主机报告 CUDA 13.0，CUDA 12.8 容器仍应正常工作。

若出现以下错误：

```text
could not select device driver "" with capabilities: [[gpu]]
```

通常表示 NVIDIA Container Toolkit 未正确安装或 Docker runtime 尚未配置。重新执行：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 5. 获取本项目分支

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs

git clone https://github.com/youngsuen19860205/indextts2.5.git
cd indextts2.5
git fetch origin
git switch copilot/create-index-tts-25-project
```

确认分支：

```bash
git branch --show-current
git log -1 --oneline
```

预期分支为：

```text
copilot/create-index-tts-25-project
```

## 6. 检查项目目录

```bash
ls -la
ls -la docker scripts indextts_batch reference gen_text gen_wav
```

主要文件：

```text
docker/Dockerfile                    Docker 镜像定义
docker/entrypoint.sh                 容器入口
Docker-compose.yml                   Docker Compose 配置（实际文件名为 docker-compose.yml）
scripts/download_models.sh           模型下载脚本
indextts_batch/cli.py                 批量生成主程序
indextts_batch/random_reference.py    随机选择 reference 的执行程序
reference/                            参考音频与同名文本
gen_text/                             待合成文本
gen_wav/                              WAV 输出目录
```

实际 Compose 文件名为：

```text
docker-compose.yml
```

创建运行所需目录：

```bash
mkdir -p checkpoints reference gen_text gen_wav
```

## 7. 构建 IndexTTS 2.5 Docker 镜像

在仓库根目录执行：

```bash
docker compose build --progress=plain
```

或：

```bash
make build
```

构建过程会：

1. 拉取 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`；
2. 安装 `ffmpeg`、`libsndfile1`、Git、编译工具和 Python；
3. 安装 `uv`；
4. 根据上游 `uv.lock` 安装 PyTorch cu128 和 IndexTTS 依赖；
5. 拷贝批量生成脚本；
6. 不把模型权重打进镜像。

首次构建需要下载较多依赖，耗时取决于网络带宽。

构建后检查镜像：

```bash
docker images | grep 'indextts2.5-batch'
```

## 8. 检查容器内 PyTorch 和 H20

```bash
docker compose run --rm indextts25 check-gpu
```

预期输出包括：

```text
CUDA 可用: True
GPU 0: NVIDIA H20 ...
```

还应检查 PyTorch CUDA 构建版本：

```bash
docker compose run --rm indextts25 python -c '
import torch
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
'
```

重点：

- `torch.version.cuda` 显示 12.8 是正常的；
- 宿主机 `nvidia-smi` 显示 CUDA 13.0 也是正常的；
- 真正的验收条件是 `torch.cuda.is_available()` 为 `True` 且能识别 H20。

## 9. 下载 IndexTTS 2.5 模型

模型默认写入宿主机的 `./checkpoints`，通过 volume 挂载到容器 `/workspace/checkpoints`。

### 9.1 Hugging Face

```bash
docker compose run --rm indextts25 download huggingface
```

等价命令：

```bash
make download SOURCE=huggingface
```

如果服务器访问 Hugging Face 较慢：

```bash
export HF_ENDPOINT=https://hf-mirror.com
docker compose run --rm indextts25 download huggingface
```

### 9.2 ModelScope

```bash
docker compose run --rm indextts25 download modelscope
```

等价命令：

```bash
make download SOURCE=modelscope
```

### 9.3 检查模型文件

```bash
ls -lh checkpoints
```

至少应包含：

```text
config.yaml
gpt.pth
s2mel.pth
codec.pth
multilingual_zh_ja_yue_char_del.tiktoken
wav2vec2bert_stats.pt
```

不要把 `checkpoints/` 提交到 Git。

## 10. 准备 reference 参考音频

每个参考样本必须由同 basename 的 WAV 和 TXT 组成：

```text
reference/
├── speaker_01.wav
├── speaker_01.txt
├── speaker_02.wav
└── speaker_02.txt
```

其中：

```text
speaker_01.wav：需要克隆的声音
speaker_01.txt：speaker_01.wav 中实际说出的内容，UTF-8 编码
```

写入参考文本示例：

```bash
printf '%s\n' '这是参考音频中实际说出的完整内容。' \
  > reference/speaker_01.txt
```

建议：

- 单人说话；
- 3–15 秒；
- 无背景音乐；
- 尽量无噪声、混响和爆音；
- 参考文本与音频内容准确对应；
- 只使用已获得授权的声音。

如果原始音频不是标准 WAV，可先转换：

```bash
ffmpeg -i input_audio.m4a \
  -ac 1 \
  -ar 24000 \
  -c:a pcm_s16le \
  reference/speaker_01.wav
```

检查文件：

```bash
file reference/speaker_01.wav
ffprobe -hide_banner reference/speaker_01.wav
cat reference/speaker_01.txt
```

## 11. 准备待合成文本

`gen_text/` 中一个 UTF-8 TXT 文件代表一条待生成音频：

```text
gen_text/
├── navigation.zh.txt
gen_text/
├── greeting.en.txt
├── notification.ja.txt
├── introduction.es.txt
└── reminder.ar.txt
```

实际目录应写成：

```text
gen_text/
├── navigation.zh.txt
├── greeting.en.txt
├── notification.ja.txt
├── introduction.es.txt
└── reminder.ar.txt
```

创建示例：

```bash
printf '%s\n' '前方三百米右转。' > gen_text/navigation.zh.txt
printf '%s\n' 'Hello, welcome to the voice assistant.' > gen_text/greeting.en.txt
printf '%s\n' 'こんにちは、音声アシスタントです。' > gen_text/notification.ja.txt
printf '%s\n' 'Hola, bienvenido al asistente de voz.' > gen_text/introduction.es.txt
printf '%s\n' 'مرحبا، أهلا بك في المساعد الصوتي.' > gen_text/reminder.ar.txt
```

语言代码：

| 后缀 | 语言 |
| --- | --- |
| `.zh.txt` | 中文 |
| `.en.txt` | 英文 |
| `.zhen.txt` | 中英文混合 |
| `.ja.txt` | 日语 |
| `.es.txt` | 西班牙语 |
| `.ar.txt` | 阿拉伯语 |

文件名语言后缀优先于命令行 `--lang`。无语言后缀的 TXT 使用 `--lang` 指定的默认语言。

## 12. 先执行 dry-run

Dry-run 不加载模型、不占用大量显存、不生成 WAV，只检查文件配对和任务计划。

指定一个 reference：

```bash
docker compose run --rm indextts25 dry-run \
  --reference speaker_01 \
  --lang zh
```

如果希望使用全部 reference：

```bash
docker compose run --rm indextts25 dry-run --lang zh
```

随机选择一个 reference：

```bash
docker compose run --rm indextts25 \
  python -m indextts_batch.random_reference \
  --reference-dir reference \
  --random-reference-count 1 \
  -- \
  --model-dir checkpoints \
  --gen-text-dir gen_text \
  --gen-wav-dir gen_wav \
  --lang zh \
  --dry-run
```

随机选择多个 reference：

```bash
docker compose run --rm indextts25 \
  python -m indextts_batch.random_reference \
  --reference-dir reference \
  --random-reference-count 3 \
  --selection-seed 20260907 \
  -- \
  --model-dir checkpoints \
  --gen-text-dir gen_text \
  --gen-wav-dir gen_wav \
  --lang zh \
  --dry-run
```

`--selection-seed` 用于复现 reference 抽样结果；不提供时，每次执行都会重新随机。

## 13. 执行脚本生成 WAV

### 13.1 指定一个 reference，生成全部目标文本

```bash
docker compose run --rm indextts25 generate \
  --reference speaker_01 \
  --lang zh \
  --device cuda:0 \
  --seed 42 \
  --continue-on-error
```

### 13.2 使用全部 reference 批量生成

```bash
docker compose run --rm indextts25 generate \
  --lang zh \
  --device cuda:0 \
  --seed 42 \
  --continue-on-error
```

如果有 `N` 个 reference 和 `M` 个文本，将生成 `N × M` 个 WAV。

### 13.3 每次随机选择一个 reference

```bash
docker compose run --rm indextts25 \
  python -m indextts_batch.random_reference \
  --reference-dir reference \
  --random-reference-count 1 \
  -- \
  --model-dir checkpoints \
  --gen-text-dir gen_text \
  --gen-wav-dir gen_wav \
  --lang zh \
  --device cuda:0 \
  --seed 42 \
  --continue-on-error
```

### 13.4 每次随机选择多个 reference

```bash
docker compose run --rm indextts25 \
  python -m indextts_batch.random_reference \
  --reference-dir reference \
  --random-reference-count 3 \
  --selection-seed 20260907 \
  -- \
  --model-dir checkpoints \
  --gen-text-dir gen_text \
  --gen-wav-dir gen_wav \
  --lang en \
  --device cuda:0 \
  --seed 42 \
  --continue-on-error
```

两个随机种子的用途不同：

| 参数 | 作用 |
| --- | --- |
| `--selection-seed` | 控制本次选择哪些 reference |
| `--seed` | 控制模型推理随机状态 |

### 13.5 使用 Makefile

指定 reference：

```bash
make generate \
  LANG_CODE=en \
  ARGS="--reference speaker_01 --device cuda:0 --seed 42 --continue-on-error"
```

当前分支的随机 reference 功能可直接通过 `python -m indextts_batch.random_reference` 调用；如果 Makefile 尚未提供 `random-generate` 目标，请使用第 13.3 或 13.4 节的 Docker Compose 命令。

## 14. 检查生成结果

```bash
find gen_wav -maxdepth 1 -type f -printf '%f\n' | sort
```

输出示例：

```text
greeting.en__speaker_01.wav
navigation.zh__speaker_01.wav
notification.ja__speaker_01.wav
manifest.jsonl
```

程序会把每个成功输出转换并验证为：

```text
格式：WAV
声道：1（mono）
采样率：16000 Hz
位深：16-bit signed PCM
```

使用 `ffprobe` 检查：

```bash
ffprobe -v error \
  -select_streams a:0 \
  -show_entries stream=codec_name,sample_rate,channels,bits_per_sample \
  -of default=noprint_wrappers=1 \
  gen_wav/greeting.en__speaker_01.wav
```

或用 Python 检查所有 WAV：

```bash
docker compose run --rm indextts25 python - <<'PY'
from pathlib import Path
import wave

for path in sorted(Path("gen_wav").glob("*.wav")):
    with wave.open(str(path), "rb") as wav:
        print(
            path,
            "channels=", wav.getnchannels(),
            "sample_rate=", wav.getframerate(),
            "bits=", wav.getsampwidth() * 8,
            "frames=", wav.getnframes(),
        )
PY
```

每个文件应显示：

```text
channels=1 sample_rate=16000 bits=16
```

查看执行清单：

```bash
cat gen_wav/manifest.jsonl
```

Manifest 包含输入文本、reference、实际语言、输出路径、状态、生成时长、推理耗时和错误信息。

## 15. 一套可复制的完整流程

以下命令假设 Docker 和 NVIDIA Container Toolkit 已安装：

```bash
# 1. 获取代码
git clone https://github.com/youngsuen19860205/indextts2.5.git
cd indextts2.5
git fetch origin
git switch copilot/create-index-tts-25-project

# 2. 创建目录
mkdir -p checkpoints reference gen_text gen_wav

# 3. 构建容器
docker compose build --progress=plain

# 4. 检查 H20
docker compose run --rm indextts25 check-gpu

# 5. 下载模型；国内网络推荐 ModelScope
docker compose run --rm indextts25 download modelscope

# 6. 放入已获授权的参考音频，然后填写对应文本
# cp /path/to/authorized_voice.wav reference/speaker_01.wav
printf '%s\n' '这里填写参考音频中实际说出的文字。' > reference/speaker_01.txt

# 7. 创建待合成文本
printf '%s\n' '前方三百米右转。' > gen_text/navigation.zh.txt
printf '%s\n' 'Hello, welcome to the voice assistant.' > gen_text/greeting.en.txt

# 8. 不加载模型，先检查任务
docker compose run --rm indextts25 dry-run \
  --reference speaker_01 \
  --lang zh

# 9. 正式生成 WAV
docker compose run --rm indextts25 generate \
  --reference speaker_01 \
  --lang zh \
  --device cuda:0 \
  --seed 42 \
  --continue-on-error

# 10. 查看输出
ls -lh gen_wav
cat gen_wav/manifest.jsonl
```

## 16. 常见问题

### 16.1 宿主机显示 CUDA 13.0，容器显示 12.8

这是预期行为。宿主机驱动支持 CUDA 13.0，而容器使用项目锁定的 CUDA 12.8 用户态 runtime。只要 PyTorch 能识别 H20，就不需要修改。

### 16.2 `torch.cuda.is_available()` 为 False

依次检查：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
docker compose run --rm indextts25 check-gpu
```

第一条失败是宿主机驱动问题；第二条失败通常是 NVIDIA Container Toolkit 或 Docker runtime 问题；前两条成功而第三条失败，才需要检查项目镜像/PyTorch。

### 16.3 Docker 构建时依赖下载失败

重新执行：

```bash
docker compose build --no-cache --progress=plain
```

如果是 Python 源访问问题，可按网络环境配置代理或镜像，但不要随意修改 `uv.lock` 中的 PyTorch CUDA 版本。

### 16.4 CUDA OOM

```bash
docker compose run --rm indextts25 generate \
  --reference speaker_01 \
  --lang zh \
  --device cuda:0 \
  --max-text-tokens-per-segment 80 \
  --continue-on-error
```

并检查其他 GPU 进程：

```bash
nvidia-smi
```

不要增加 `--fp32`；默认 bf16 更节省显存。

### 16.5 模型目录不完整

```bash
rm -rf checkpoints/*
docker compose run --rm indextts25 download modelscope
```

然后检查：

```bash
ls -lh checkpoints
```

### 16.6 reference 配对失败

确认严格同名：

```text
speaker_01.wav
speaker_01.txt
```

以下写法不能配对：

```text
speaker_01.wav
speaker-01.txt
```

### 16.7 输出文件权限问题

Compose 容器默认可能以 root 写入挂载目录。可以先修复目录所有者：

```bash
sudo chown -R "$USER":"$USER" checkpoints reference gen_text gen_wav
```

## 17. 验收清单

逐项执行：

```bash
nvidia-smi
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
docker compose build
docker compose run --rm indextts25 check-gpu
docker compose run --rm indextts25 dry-run --reference speaker_01 --lang zh
docker compose run --rm indextts25 generate --reference speaker_01 --lang zh --device cuda:0
```

最终确认：

- H20 在宿主机和容器中均可见；
- PyTorch CUDA 可用；
- 模型文件完整；
- reference WAV/TXT 配对正确；
- `gen_text/` 中至少有一个非空 UTF-8 TXT；
- `gen_wav/` 中生成 WAV；
- WAV 为 mono、16000 Hz、16-bit PCM；
- `manifest.jsonl` 中任务状态为 `ok`。
