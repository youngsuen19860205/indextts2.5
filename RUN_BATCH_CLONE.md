# IndexTTS 2.5 批量语音克隆执行说明

本文档单独说明本仓库新增功能的位置、输入输出目录，以及如何在 NVIDIA H20 + Ubuntu + Docker 环境中执行 IndexTTS 2.5 批量语音克隆。

> 合成语言不固定为中文。可以通过命令行 `--lang` 动态设置默认语言，也可以通过待合成文本的文件名为每条任务单独指定语言。

## 1. 实现代码位置

| 功能 | 文件或目录 |
| --- | --- |
| 批量推理命令行入口 | `indextts_batch/cli.py` |
| `python -m indextts_batch` 入口 | `indextts_batch/__main__.py` |
| 参考音频/文本配对、任务发现、输出命名 | `indextts_batch/tasks.py` |
| 输出音频转为 16 kHz、16-bit、单声道 WAV | `indextts_batch/audio.py` |
| IndexTTS 2.5 官方推理实现 | `indextts/infer_v2_5.py` |
| H20 CUDA Docker 镜像 | `docker/Dockerfile` |
| 容器命令入口 | `docker/entrypoint.sh` |
| Docker Compose 配置 | `docker-compose.yml` |
| 模型下载脚本 | `scripts/download_models.sh` |
| 常用命令 | `Makefile` |
| 单元测试 | `tests/test_batch_clone.py` |

批量程序调用官方模型接口：

```python
from indextts.infer_v2_5 import IndexTTS2
```

模型在一个批次中只加载一次，然后依次处理 `gen_text/` 中的文本文件。

## 2. 获取 PR #1 的代码

在 PR #1 合并前，可直接检出对应分支：

```bash
git clone https://github.com/youngsuen19860205/indextts2.5.git
cd indextts2.5
git fetch origin
git switch copilot/create-index-tts-25-project
```

如果已安装 GitHub CLI，也可以执行：

```bash
gh pr checkout 1
```

## 3. 支持的语言与动态设置

当前 IndexTTS 2.5 批量程序支持以下语言代码：

| 代码 | 语言 |
| --- | --- |
| `zh` | 中文 |
| `en` | 英文 |
| `zhen` | 中英文混合 |
| `ja` | 日语 |
| `es` | 西班牙语 |
| `ar` | 阿拉伯语 |

语言有两种设置方式。

### 3.1 为整个批次设置默认语言

通过 `--lang` 动态指定：

```bash
docker compose run --rm indextts25 generate --lang en
docker compose run --rm indextts25 generate --lang ja
docker compose run --rm indextts25 generate --lang es
docker compose run --rm indextts25 generate --lang ar
```

使用 Makefile 时通过 `LANG_CODE` 设置：

```bash
make generate LANG_CODE=en
make generate LANG_CODE=ja
make generate LANG_CODE=es
make generate LANG_CODE=ar
```

`LANG_CODE` 未设置时默认值为 `zh`，但这只是默认值，不限制实际合成语言。

### 3.2 为单个文本文件设置语言

文本文件可以使用以下命名格式：

```text
<名称>.<语言代码>.txt
```

例如：

```text
gen_text/welcome.zh.txt
gen_text/welcome_en.en.txt
gen_text/welcome_mix.zhen.txt
gen_text/welcome_ja.ja.txt
gen_text/welcome_es.es.txt
gen_text/welcome_ar.ar.txt
```

文件名中的语言代码优先级高于命令行的 `--lang`。因此下面的命令可以一次处理多种语言：

```bash
docker compose run --rm indextts25 generate --lang zh
```

其中：

- `welcome.zh.txt` 使用中文；
- `welcome_en.en.txt` 使用英文；
- `welcome_mix.zhen.txt` 使用中英文混合；
- `welcome_ja.ja.txt` 使用日语；
- `welcome_es.es.txt` 使用西班牙语；
- `welcome_ar.ar.txt` 使用阿拉伯语；
- 没有语言后缀的文件使用 `--lang` 指定的默认语言。

这意味着参考音频的语言与目标文本语言可以不同，可用于跨语种音色克隆。

## 4. 准备参考数据

在 `reference/` 中放置参考 WAV 和同名 UTF-8 文本：

```text
reference/
├── speaker_01.wav
└── speaker_01.txt
```

例如：

```text
reference/speaker_01.wav
reference/speaker_01.txt
```

`speaker_01.txt` 应填写 `speaker_01.wav` 中实际说出的内容。

要求与建议：

- WAV 和 TXT 的 basename 必须完全相同；
- 建议参考音频时长为 3–15 秒；
- 建议只包含一个说话人；
- 避免背景音乐、明显噪声和混响；
- 参考文本使用 UTF-8 编码；
- 只使用已获得授权的声音。

如果有多个参考说话人，可以分别准备：

```text
reference/
├── speaker_01.wav
├── speaker_01.txt
├── speaker_02.wav
└── speaker_02.txt
```

默认情况下程序会使用全部有效 reference。也可以通过 `--reference` 选择一个或多个：

```bash
docker compose run --rm indextts25 generate \
  --reference speaker_01 \
  --lang en
```

选择多个 reference：

```bash
docker compose run --rm indextts25 generate \
  --reference speaker_01 \
  --reference speaker_02 \
  --lang en
```

## 5. 准备待合成文本

`gen_text/` 中一个 UTF-8 `.txt` 文件表示一条合成任务：

```text
gen_text/
├── greeting.en.txt
├── navigation.zh.txt
├── notification.ja.txt
├── introduction.es.txt
└── reminder.ar.txt
```

示例：

```bash
printf '%s\n' 'Hello, welcome to the voice assistant.' > gen_text/greeting.en.txt
printf '%s\n' '前方三百米右转。' > gen_text/navigation.zh.txt
printf '%s\n' 'こんにちは、音声アシスタントです。' > gen_text/notification.ja.txt
printf '%s\n' 'Hola, bienvenido al asistente de voz.' > gen_text/introduction.es.txt
printf '%s\n' 'مرحبا، أهلا بك في المساعد الصوتي.' > gen_text/reminder.ar.txt
```

## 6. 构建 Docker 镜像

宿主机应已安装：

- NVIDIA 驱动；
- Docker；
- NVIDIA Container Toolkit。

镜像内包含 CUDA 12.8 运行环境。宿主机不需要安装相同版本的 CUDA Toolkit，但 NVIDIA 驱动必须与容器 CUDA 版本兼容。

构建镜像：

```bash
make build
```

等价命令：

```bash
docker compose build
```

检查 H20 GPU 是否可用：

```bash
make check-gpu
```

输出中应确认：

- `nvidia-smi` 能识别 H20；
- `torch.cuda.is_available()` 为 `True`；
- PyTorch 能获取 CUDA 设备；
- GPU 架构为 Hopper / `sm_90`。

## 7. 下载 IndexTTS 2.5 模型

使用 Hugging Face：

```bash
make download
```

国内网络可以使用 ModelScope：

```bash
make download SOURCE=modelscope
```

也可以直接执行：

```bash
scripts/download_models.sh huggingface ./checkpoints
scripts/download_models.sh modelscope ./checkpoints
```

下载完成后，模型文件位于 `checkpoints/`。模型权重不会被打包进 Docker 镜像。

## 8. Dry-run 检查

正式推理前建议执行 dry-run。该操作不会加载模型，也不会生成音频，只检查：

- reference WAV/TXT 是否正确配对；
- 文本文件是否为空；
- 每条文本解析出的语言；
- reference、文本和输出文件之间的任务映射；
- 输出文件名是否冲突。

使用默认语言：

```bash
make dry-run LANG_CODE=en
```

或直接传入动态语言：

```bash
docker compose run --rm indextts25 dry-run \
  --reference speaker_01 \
  --lang en
```

如果文本文件名带有语言后缀，dry-run 输出中的 `lang` 会显示对应文件的实际语言，而不是统一使用默认语言。

## 9. 正式批量生成

### 9.1 单一默认语言

```bash
make generate LANG_CODE=en \
  ARGS="--reference speaker_01 --seed 42 --continue-on-error"
```

### 9.2 多语言混合批次

将语言写入各文本文件名后执行：

```bash
docker compose run --rm indextts25 generate \
  --reference speaker_01 \
  --lang zh \
  --seed 42 \
  --continue-on-error
```

`--lang zh` 只作为无语言后缀文件的默认值。带 `.en.txt`、`.ja.txt`、`.es.txt`、`.ar.txt` 或 `.zhen.txt` 后缀的文件会动态选择自己的语言。

### 9.3 直接使用 Python CLI

如果已经进入容器或正确安装了项目依赖：

```bash
python -m indextts_batch \
  --model-dir checkpoints \
  --reference-dir reference \
  --gen-text-dir gen_text \
  --gen-wav-dir gen_wav \
  --reference speaker_01 \
  --lang en \
  --device cuda:0 \
  --seed 42 \
  --continue-on-error
```

查看完整参数：

```bash
python -m indextts_batch --help
```

## 10. 输出结果

生成结果写入 `gen_wav/`：

```text
gen_wav/
├── greeting.en__speaker_01.wav
├── navigation.zh__speaker_01.wav
├── notification.ja__speaker_01.wav
├── introduction.es__speaker_01.wav
├── reminder.ar__speaker_01.wav
└── manifest.jsonl
```

每个成功生成的 WAV 都会被转换并验证为：

- WAV；
- 单声道；
- 16000 Hz；
- signed 16-bit PCM；
- 非空音频。

`manifest.jsonl` 记录：

- 输入文本文件和文本内容；
- reference 名称、音频路径和参考文本；
- 每条任务实际使用的语言；
- 输出 WAV 路径；
- 执行状态；
- 生成音频时长；
- 推理耗时；
- 错误信息。

## 11. 常用参数

| 参数 | 说明 |
| --- | --- |
| `--lang` | 动态设置无语言后缀文本的默认语言 |
| `--reference NAME` | 选择 reference basename，可重复指定 |
| `--model-dir` | 模型目录，默认 `checkpoints` |
| `--reference-dir` | reference 目录，默认 `reference` |
| `--gen-text-dir` | 待合成文本目录，默认 `gen_text` |
| `--gen-wav-dir` | 输出目录，默认 `gen_wav` |
| `--device` | `auto`、`cuda`、`cuda:0` 或 `cpu` |
| `--seed` | 随机种子 |
| `--duration-factor` | 语速/时长缩放因子 |
| `--emo-alpha` | 情感强度 |
| `--continue-on-error` | 单条失败后继续处理其他任务 |
| `--skip-existing` | 跳过已经存在的输出文件 |
| `--dry-run` | 只检查任务，不加载模型 |
| `--keep-raw` | 保留模型原始采样率输出 |

## 12. 最短执行流程

```bash
# 1. 切换到 PR #1 分支
git switch copilot/create-index-tts-25-project

# 2. 准备 reference/speaker_01.wav 和 reference/speaker_01.txt
# 3. 准备 gen_text/*.txt，可通过 .<lang>.txt 动态指定语言

# 4. 构建镜像并检查 GPU
make build
make check-gpu

# 5. 下载模型
make download SOURCE=modelscope

# 6. 检查任务
make dry-run LANG_CODE=en ARGS="--reference speaker_01"

# 7. 批量生成
make generate LANG_CODE=en \
  ARGS="--reference speaker_01 --seed 42 --continue-on-error"
```

如果需要在同一批次生成多种语言，请为文本使用语言后缀，然后执行一次 generate：

```bash
# gen_text/a.zh.txt
# gen_text/b.en.txt
# gen_text/c.ja.txt
# gen_text/d.es.txt
# gen_text/e.ar.txt

docker compose run --rm indextts25 generate \
  --reference speaker_01 \
  --lang zh \
  --continue-on-error
```

## 13. 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 所有任务成功，或 dry-run 校验成功 |
| `1` | 至少一条生成任务失败 |
| `2` | 输入、模型目录、依赖或设备校验失败 |
