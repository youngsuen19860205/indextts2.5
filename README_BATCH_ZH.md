# IndexTTS 2.5 批量语音克隆（H20 + Ubuntu + Docker）

本项目在 [index-tts/index-tts](https://github.com/index-tts/index-tts) 官方代码之上，
提供一个**可落地运行的批量零样本 / 跨语种语音克隆流水线**：

- 从 `reference/` 读取参考音频（`*.wav`）及其同名参考文本（`*.txt`）；
- 从 `gen_text/` 逐文件读取待合成文本（一个 `.txt` 一条文本）；
- 调用官方 IndexTTS 2.5 推理类 `indextts.infer_v2_5.IndexTTS2`，模型**只加载一次**批量生成；
- 输出到 `gen_wav/`，并保证每个文件都是**单声道 / 16000 Hz / 16-bit PCM WAV**（写出后逐个校验）；
- 输出 `manifest.jsonl`，记录输入文本、参考说话人、输出路径、状态、错误与时长。

上游依据版本：本仓库基于上游 commit
[`ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c`](https://github.com/index-tts/index-tts/commit/ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c)
（包含 `indextts/infer_v2_5.py` 与 IndexTTS 2.5 支持）。所使用的 API 均来自该版本：

| 用途 | 上游 API / 事实 |
| --- | --- |
| 加载模型 | `IndexTTS2(cfg_path, model_dir, use_bf16, device, use_cuda_kernel, use_deepspeed, use_qwen_emo)` |
| 单条合成 | `IndexTTS2.infer(spk_audio_prompt, text, output_path, lang, emo_alpha, use_random, interval_silence, verbose, max_text_tokens_per_segment, duration_factor, text_normalization)` |
| 配置文件 | `<model_dir>/config.yaml`（`indextts/utils/model_download.ensure_config_available`） |
| 权重仓库 | `IndexTeam/IndexTTS-2.5`（Hugging Face / ModelScope） |
| 必需权重文件 | `gpt.pth`、`s2mel.pth`、`codec.pth`、`multilingual_zh_ja_yue_char_del.tiktoken`、`wav2vec2bert_stats.pt` |
| 支持语种 | `zh`、`en`、`zhen`、`ja`、`es`、`ar`（见 `infer_v2_5.py` 中的 `lang` 分支） |

> 依赖版本由上游 `pyproject.toml` / `uv.lock` 锁定（PyTorch 2.8 + CUDA 12.8），
> 因此不再额外固定 commit，如需切换上游版本可自行 rebase 并重新构建镜像。

## 1. 目录结构

```
.
├── docker/Dockerfile          # CUDA 12.8 + PyTorch(cu128) 运行环境
├── docker/entrypoint.sh       # 容器入口：check-gpu / download / dry-run / generate / shell
├── docker-compose.yml         # 挂载模型、数据目录与 HF/ModelScope 缓存
├── Makefile                   # build / download / dry-run / generate / shell / check-gpu / test / lint
├── scripts/download_models.sh # 下载 IndexTTS 2.5 权重（HF 或 ModelScope）
├── indextts_batch/            # 批量推理程序（本项目新增）
│   ├── tasks.py               # reference/gen_text 配对、任务规划、输出命名
│   ├── audio.py               # 16kHz/16bit/mono 后处理与格式校验
│   └── cli.py                 # 命令行入口（python -m indextts_batch）
├── indextts/                  # 上游官方模型与推理代码
├── reference/                 # 参考音频 + 同名参考文本（见 reference/README.md）
├── gen_text/                  # 待合成文本，一个文件一条（见 gen_text/README.md）
├── gen_wav/                   # 生成结果 + manifest.jsonl（保留 .gitkeep）
└── tests/test_batch_clone.py  # 无需 GPU / 无需模型的单元测试
```

## 2. 前置条件（NVIDIA H20 + Ubuntu）

1. Ubuntu 20.04/22.04，NVIDIA H20（Hopper，`sm_90`）。
2. 宿主机 NVIDIA 驱动 ≥ 525（H20 推荐 535+），`nvidia-smi` 可正常输出。
3. Docker 与 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

> **重要**：镜像内已包含 CUDA 12.8 运行库，宿主机**只需要兼容的 NVIDIA 驱动**，
> **不需要**在宿主机安装相同版本的 CUDA Toolkit。

## 3. 构建镜像

```bash
git clone https://github.com/youngsuen19860205/indextts2.5.git
cd indextts2.5
make build                     # 等价于 docker compose build
make check-gpu                 # 打印 nvidia-smi / torch 版本 / CUDA 可用性 / GPU 型号
```

不使用 compose 时：

```bash
docker build -t indextts2.5-batch:latest -f docker/Dockerfile .
docker run --rm --gpus all indextts2.5-batch:latest check-gpu
```

## 4. 下载 IndexTTS 2.5 权重

权重不会打进镜像，运行时挂载 `./checkpoints`：

```bash
make download                  # Hugging Face
make download SOURCE=modelscope  # 国内建议使用 ModelScope
```

或在宿主机直接执行（国内可先 `export HF_ENDPOINT=https://hf-mirror.com`）：

```bash
scripts/download_models.sh huggingface ./checkpoints
scripts/download_models.sh modelscope  ./checkpoints
```

脚本会校验 `gpt.pth`、`s2mel.pth`、`codec.pth`、`multilingual_zh_ja_yue_char_del.tiktoken`、
`wav2vec2bert_stats.pt`、`config.yaml` 是否齐全。

## 5. 准备数据

`reference/`（参考音色，UTF-8 文本，basename 必须一致）：

```
reference/speaker_zh.wav
reference/speaker_zh.txt      # 该音频里实际说出的内容
```

`gen_text/`（待合成文本，一个文件一条，可用 `.<lang>.txt` 指定语种）：

```
gen_text/hello_zh.txt         # 默认使用 --lang
gen_text/hello_en.en.txt      # 英文
gen_text/hello_ja.ja.txt      # 日语
```

## 6. 批量生成

先做不加载模型的配对/计划校验：

```bash
make dry-run
# 或：docker compose run --rm indextts25 dry-run --lang zh
# 或宿主机直接：python -m indextts_batch --dry-run
```

正式生成（模型只加载一次，逐个处理 `gen_text/*.txt`）：

```bash
make generate LANG_CODE=zh
# 传递更多参数：
make generate ARGS="--reference speaker_zh --seed 42 --continue-on-error"
```

等价的 `docker run` 写法：

```bash
docker run --rm --gpus all \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  -v "$PWD/reference:/workspace/reference" \
  -v "$PWD/gen_text:/workspace/gen_text" \
  -v "$PWD/gen_wav:/workspace/gen_wav" \
  -v indextts-cache:/cache \
  -e HF_ENDPOINT=https://hf-mirror.com \
  indextts2.5-batch:latest generate --lang zh --seed 42
```

跨语种示例（用中文参考音色朗读英文 / 日语文本）：

```bash
printf '%s\n' 'Cross-lingual cloning keeps the Chinese speaker timbre.' > gen_text/cross_en.en.txt
docker compose run --rm indextts25 generate --reference speaker_zh --lang zh
```

（`cross_en.en.txt` 的 `.en` 后缀会覆盖 `--lang zh`，其余文件仍按 `--lang` 处理。）

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--model-dir` | 权重目录，默认 `checkpoints`（或环境变量 `INDEXTTS_MODEL_DIR`） |
| `--cfg-path` | 配置文件，默认 `<model-dir>/config.yaml` |
| `--reference-dir` / `--gen-text-dir` / `--gen-wav-dir` | 三个数据目录 |
| `--reference NAME` | 指定参考说话人 basename，可重复；默认使用全部（按文件名排序） |
| `--name-template` | 输出命名模板，默认 `{text}__{reference}`，可改为 `{text}` |
| `--lang` | 默认语种：`zh` `en` `zhen` `ja` `es` `ar` |
| `--device` | `auto`（默认）/ `cuda:0` / `cpu`；请求 CUDA 但不可用时直接报错 |
| `--seed` | 随机种子，默认 `0`，每条合成前重置以保证可复现 |
| `--fp32` | 关闭 bf16（默认在 CUDA 上使用 bf16） |
| `--emo-alpha` / `--interval-silence` / `--max-text-tokens-per-segment` / `--duration-factor` / `--no-text-normalization` | 透传给上游 `infer()` 的推理参数 |
| `--manifest` | manifest 路径，默认 `<gen-wav-dir>/manifest.jsonl`，`.csv` 后缀写 CSV |
| `--skip-existing` / `--keep-raw` / `--continue-on-error` / `--dry-run` / `--verbose` | 其他控制项 |

## 7. 输出格式保证

上游模型输出为 22.05 kHz，本项目在每条合成后：

1. 用 `soundfile` 读取原始输出、做单声道混合、用 `soxr`/`librosa` 重采样到 16000 Hz，
   再以 `PCM_16` 子类型写出 WAV；Python 库不可用时自动回退到 `ffmpeg`；
2. 用标准库 `wave` **重新读取并校验** 声道数=1、采样率=16000、位深=16-bit、非空；
3. 校验失败即视为该任务失败，写入 manifest 的 `error` 字段。

`gen_wav/manifest.jsonl` 每行示例：

```json
{"text_file": "gen_text/hello_zh.txt", "text": "你好…", "reference": "speaker_zh",
 "reference_wav": "reference/speaker_zh.wav", "reference_text": "…", "language": "zh",
 "output_wav": "gen_wav/hello_zh__speaker_zh.wav", "status": "ok", "sample_rate": 16000,
 "duration_seconds": 3.52, "elapsed_seconds": 4.1, "error": ""}
```

退出码：`0` 全部成功；`1` 存在失败任务（配合 `--continue-on-error` 会跑完全部任务再返回 `1`）；`2` 输入/环境校验失败。

可用 `ffprobe` 或 Python 二次确认：

```bash
python - <<'PY'
import wave
with wave.open("gen_wav/hello_zh__speaker_zh.wav") as w:
    print(w.getnchannels(), w.getframerate(), w.getsampwidth() * 8)
PY
# 输出：1 16000 16
```

## 8. 克隆效果与合规提醒

- 参考音频建议 3–15 秒、单人、无背景音乐/噪声/混响；音质越干净，音色相似度越高。
- 参考文本应与音频内容一致：虽然 IndexTTS 2.5 的 `infer()` 只需要参考音频，
  准确的参考文本有助于筛选素材与排查问题，并会写入 manifest。
- 过短（<2 秒）或过长（>30 秒）的参考音频会触发警告，通常会降低稳定性。
- **仅可克隆已获得授权的声音**；请遵守 `DISCLAIMER` 与模型许可，不得用于伪造、欺诈或侵犯他人权益。

## 9. 故障排查

| 现象 | 处理 |
| --- | --- |
| `torch.cuda.is_available() 为 False` | 确认 `--gpus all`、宿主机驱动 ≥525、已安装并配置 NVIDIA Container Toolkit（`sudo nvidia-ctk runtime configure --runtime=docker`） |
| `CUDA error: no kernel image is available` | 镜像 PyTorch 与 GPU 架构不匹配；本镜像使用 cu128（支持 H20 `sm_90`），请勿替换为旧版 torch |
| CUDA OOM | 使用默认 bf16（不要加 `--fp32`）、降低 `--max-text-tokens-per-segment`（如 80）、拆分过长文本、确认 GPU 上无其他进程 |
| 模型目录不完整 | 重新执行 `scripts/download_models.sh`，或改用 `SOURCE=modelscope` |
| 下载缓慢/失败 | `export HF_ENDPOINT=https://hf-mirror.com`，或使用 ModelScope；缓存已持久化到 `hf-cache` 卷 |
| `无法读取音频` / 参考音频解析失败 | 用 `ffmpeg -i in.m4a -ac 1 -ar 24000 reference/spk.wav` 转成标准 WAV |
| 输出不是 16 kHz | 不会发生：写出后强制校验；若报“输出音频格式校验失败”，检查 `soundfile`/`soxr`/`ffmpeg` 是否可用 |
| 依赖冲突 | 请使用镜像内的 uv 环境；宿主机手动 `pip install` 不受支持（见上游文档） |

## 10. 本地开发（无需 GPU）

```bash
python -m pip install pytest ruff numpy soundfile soxr
make test     # pytest tests/test_batch_clone.py
make lint     # ruff check + ruff format --check
python -m indextts_batch --dry-run   # 只校验配对与任务规划
```

GitHub Actions（`.github/workflows/batch-clone-ci.yml`）在无 GPU 的 runner 上执行同样的
lint 与单元测试，不下载模型、不做真实推理。
