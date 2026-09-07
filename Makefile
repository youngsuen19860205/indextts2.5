# IndexTTS 2.5 批量语音克隆常用操作
#
#   make build       构建 Docker 镜像
#   make check-gpu   验证容器内 CUDA / PyTorch / GPU
#   make download    下载 IndexTTS 2.5 权重到 ./checkpoints
#   make dry-run     只校验 reference/gen_text 配对并打印任务计划
#   make generate    批量生成 16kHz/16bit/mono WAV 到 ./gen_wav
#   make shell       进入容器交互式 shell
#   make test / make lint  本地无 GPU 的单元测试与静态检查

SHELL := /bin/bash
COMPOSE ?= docker compose
SERVICE ?= indextts25
SOURCE ?= huggingface
LANG_CODE ?= zh
ARGS ?=

.PHONY: build check-gpu download dry-run generate shell test lint format help

help:
	@grep -E '^#   ' $(MAKEFILE_LIST) | sed 's/^#   //'

build:
	$(COMPOSE) build

check-gpu:
	$(COMPOSE) run --rm $(SERVICE) check-gpu

download:
	$(COMPOSE) run --rm $(SERVICE) download $(SOURCE)

dry-run:
	$(COMPOSE) run --rm $(SERVICE) dry-run --lang $(LANG_CODE) $(ARGS)

generate:
	$(COMPOSE) run --rm $(SERVICE) generate --lang $(LANG_CODE) $(ARGS)

shell:
	$(COMPOSE) run --rm $(SERVICE) shell

test:
	python -m pytest tests/test_batch_clone.py -v

lint:
	python -m ruff check indextts_batch tests/test_batch_clone.py
	python -m ruff format --check indextts_batch tests/test_batch_clone.py

format:
	python -m ruff format indextts_batch tests/test_batch_clone.py
