"""CLI: batch zero-shot / cross-lingual voice cloning with IndexTTS 2.5.

Usage::

    python -m indextts_batch --model-dir checkpoints --lang zh

Heavy dependencies (torch, the IndexTTS model code) are imported lazily so
that ``--dry-run`` works on machines without a GPU or model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from indextts_batch import OUTPUT_SAMPLE_RATE
from indextts_batch.audio import (
    AudioProcessingError,
    convert_to_output_format,
    probe_reference_audio,
)
from indextts_batch.tasks import (
    SUPPORTED_LANGUAGES,
    BatchInputError,
    SynthesisTask,
    discover_references,
    discover_texts,
    plan_tasks,
    resolve_language,
)

#: Weight files shipped by ``IndexTeam/IndexTTS-2.5`` (see webui.py REQUIRED_FILES).
REQUIRED_MODEL_FILES = (
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "wav2vec2bert_stats.pt",
)

MIN_REFERENCE_SECONDS = 2.0
MAX_REFERENCE_SECONDS = 30.0


@dataclass
class TaskResult:
    """One manifest row."""

    text_file: str
    text: str
    reference: str
    reference_wav: str
    reference_text: str
    language: str
    output_wav: str
    status: str
    sample_rate: int = OUTPUT_SAMPLE_RATE
    duration_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    error: str = ""
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        row = {
            "text_file": self.text_file,
            "text": self.text,
            "reference": self.reference,
            "reference_wav": self.reference_wav,
            "reference_text": self.reference_text,
            "language": self.language,
            "output_wav": self.output_wav,
            "status": self.status,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 3),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "error": self.error,
        }
        row.update(self.extra)
        return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m indextts_batch",
        description=(
            "使用 IndexTTS 2.5 对 gen_text/ 中的文本做批量语音克隆，输出 16kHz/16bit/单声道 WAV。"
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("INDEXTTS_MODEL_DIR", "checkpoints"),
        help="IndexTTS 2.5 权重目录（默认 checkpoints，可用环境变量 INDEXTTS_MODEL_DIR 覆盖）",
    )
    parser.add_argument(
        "--cfg-path",
        default=None,
        help="config.yaml 路径（默认 <model-dir>/config.yaml）",
    )
    parser.add_argument("--reference-dir", default="reference", help="参考音频/文本目录")
    parser.add_argument("--gen-text-dir", default="gen_text", help="待合成文本目录")
    parser.add_argument("--gen-wav-dir", default="gen_wav", help="生成音频输出目录")
    parser.add_argument(
        "--reference",
        action="append",
        default=None,
        metavar="NAME",
        help="仅使用指定 basename 的参考说话人，可重复；默认使用全部参考（按文件名排序）",
    )
    parser.add_argument(
        "--name-template",
        default="{text}__{reference}",
        help="输出文件名模板，支持 {text} 与 {reference}（默认 {text}__{reference}）",
    )
    parser.add_argument(
        "--lang",
        default="zh",
        choices=SUPPORTED_LANGUAGES,
        help="默认合成语种；gen_text 中的 <name>.<lang>.txt 可覆盖该值",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备：auto / cuda / cuda:0 / cpu（默认 auto）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument(
        "--use-random",
        action="store_true",
        help="启用上游 infer(use_random=True) 的随机采样（默认关闭以保证可复现）",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="关闭 bf16 推理（默认在 CUDA 上使用 bf16）",
    )
    parser.add_argument(
        "--emo-alpha", type=float, default=1.0, help="情感强度 emo_alpha（默认 1.0）"
    )
    parser.add_argument(
        "--interval-silence",
        type=int,
        default=200,
        help="分段之间插入的静音毫秒数（默认 200）",
    )
    parser.add_argument(
        "--max-text-tokens-per-segment",
        type=int,
        default=120,
        help="每段最大 token 数（默认 120）",
    )
    parser.add_argument(
        "--duration-factor", type=float, default=1.0, help="时长缩放因子（默认 1.0）"
    )
    parser.add_argument(
        "--no-text-normalization",
        action="store_true",
        help="关闭上游文本正则化",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="manifest 输出路径（默认 <gen-wav-dir>/manifest.jsonl；.csv 后缀写 CSV）",
    )
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在的输出文件")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="保留模型原始输出（22.05kHz）到 <gen-wav-dir>/raw/",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单条失败时继续处理其余任务，结束时返回非零状态",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验配对并打印任务计划，不加载模型",
    )
    parser.add_argument("--verbose", action="store_true", help="打印上游推理详细日志")
    return parser


def _resolve_device(requested: str) -> str:
    """Resolve ``auto`` and validate CUDA availability (imports torch lazily)."""
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        print(">> 未检测到 CUDA，回退到 CPU（速度会非常慢）", file=sys.stderr)
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise BatchInputError(
            "请求了 CUDA 设备，但 torch.cuda.is_available() 为 False。"
            "请确认已安装 NVIDIA 驱动，并使用 --gpus all 启动容器。"
        )
    return requested


def _set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency of torch stack
        pass


def _validate_model_dir(model_dir: Path, cfg_path: Path) -> None:
    if not model_dir.is_dir():
        raise BatchInputError(f"模型目录不存在: {model_dir}，请先运行 scripts/download_models.sh")
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if not cfg_path.is_file():
        missing.append(str(cfg_path))
    if missing:
        raise BatchInputError(
            f"模型目录 {model_dir} 不完整，缺少: {', '.join(sorted(set(missing)))}。"
            "请运行 scripts/download_models.sh 下载 IndexTeam/IndexTTS-2.5。"
        )


def _validate_references(tasks: list[SynthesisTask]) -> None:
    checked: set[Path] = set()
    for task in tasks:
        wav_path = task.reference.wav_path
        if wav_path in checked:
            continue
        checked.add(wav_path)
        info = probe_reference_audio(wav_path)
        duration = info.duration_seconds
        if duration <= 0:
            raise AudioProcessingError(f"参考音频为空: {wav_path}")
        if duration < MIN_REFERENCE_SECONDS or duration > MAX_REFERENCE_SECONDS:
            print(
                f">> 警告: 参考音频 {wav_path.name} 时长 {duration:.1f}s，"
                f"建议使用 {MIN_REFERENCE_SECONDS:.0f}-{MAX_REFERENCE_SECONDS:.0f}s 的干净录音",
                file=sys.stderr,
            )


def _write_manifest(path: Path, results: list[TaskResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.as_row() for result in results]
    if path.suffix.lower() == ".csv":
        fieldnames = list(rows[0]) if rows else ["text_file", "status"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_model(args, device: str):
    """Load the official IndexTTS 2.5 inference class exactly once."""
    from indextts.infer_v2_5 import IndexTTS2

    cfg_path = Path(args.cfg_path or Path(args.model_dir) / "config.yaml")
    print(f">> 加载 IndexTTS 2.5 模型: model_dir={args.model_dir}, cfg={cfg_path}, device={device}")
    return IndexTTS2(
        cfg_path=str(cfg_path),
        model_dir=str(args.model_dir),
        use_bf16=not args.fp32 and device.startswith("cuda"),
        device=device,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_qwen_emo=False,
    )


def _describe_plan(tasks: list[SynthesisTask], default_lang: str) -> None:
    print(f">> 共规划 {len(tasks)} 个合成任务:")
    for index, task in enumerate(tasks, start=1):
        lang = resolve_language(task.text_task.name, default_lang)
        print(
            f"  [{index}] text={task.text_task.text_path} "
            f"reference={task.reference.wav_path.name} lang={lang} -> {task.output_path}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    reference_dir = Path(args.reference_dir)
    gen_text_dir = Path(args.gen_text_dir)
    gen_wav_dir = Path(args.gen_wav_dir)
    model_dir = Path(args.model_dir)
    cfg_path = Path(args.cfg_path) if args.cfg_path else model_dir / "config.yaml"
    manifest_path = Path(args.manifest) if args.manifest else gen_wav_dir / "manifest.jsonl"

    try:
        references = discover_references(reference_dir, args.reference)
        texts = discover_texts(gen_text_dir)
        tasks = plan_tasks(references, texts, gen_wav_dir, args.name_template)
        _validate_references(tasks)
    except (BatchInputError, AudioProcessingError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        _describe_plan(tasks, args.lang)
        print(">> --dry-run: 未加载模型，也未生成音频。")
        return 0

    try:
        _validate_model_dir(model_dir, cfg_path)
        device = _resolve_device(args.device)
    except BatchInputError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"错误: 无法导入 PyTorch，请检查运行环境: {exc}", file=sys.stderr)
        return 2

    gen_wav_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = gen_wav_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)
    tts = _load_model(args, device)

    results: list[TaskResult] = []
    failures = 0
    for index, task in enumerate(tasks, start=1):
        language = resolve_language(task.text_task.name, args.lang)
        result = TaskResult(
            text_file=str(task.text_task.text_path),
            text=task.text_task.text,
            reference=task.reference.name,
            reference_wav=str(task.reference.wav_path),
            reference_text=task.reference.text,
            language=language,
            output_wav=str(task.output_path),
            status="pending",
        )

        if args.skip_existing and task.output_path.is_file():
            result.status = "skipped"
            results.append(result)
            print(f"[{index}/{len(tasks)}] 已存在，跳过 {task.output_path}")
            continue

        started = time.time()
        raw_path = raw_dir / f"{task.output_path.stem}.raw.wav"
        try:
            _set_seed(args.seed)
            tts.infer(
                spk_audio_prompt=str(task.reference.wav_path),
                text=task.text_task.text,
                output_path=str(raw_path),
                lang=language.upper(),
                emo_alpha=args.emo_alpha,
                use_random=args.use_random,
                interval_silence=args.interval_silence,
                verbose=args.verbose,
                max_text_tokens_per_segment=args.max_text_tokens_per_segment,
                duration_factor=args.duration_factor,
                text_normalization=not args.no_text_normalization,
            )
            info = convert_to_output_format(raw_path, task.output_path)
            result.status = "ok"
            result.duration_seconds = info.duration_seconds
            result.sample_rate = info.sample_rate
            print(
                f"[{index}/{len(tasks)}] 生成成功 {task.output_path} "
                f"({info.duration_seconds:.2f}s, {info.sample_rate} Hz, 16-bit, mono)"
            )
        except Exception as exc:  # noqa: BLE001 - report and optionally continue
            failures += 1
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            print(
                f"[{index}/{len(tasks)}] 生成失败 {task.output_path}: {result.error}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                result.elapsed_seconds = time.time() - started
                results.append(result)
                _write_manifest(manifest_path, results)
                if not args.keep_raw:
                    shutil.rmtree(raw_dir, ignore_errors=True)
                return 1
        finally:
            if not args.keep_raw:
                raw_path.unlink(missing_ok=True)

        result.elapsed_seconds = time.time() - started
        results.append(result)

    if not args.keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)

    _write_manifest(manifest_path, results)
    ok = sum(1 for result in results if result.status == "ok")
    print(f">> 完成: 成功 {ok} / 共 {len(results)}，失败 {failures}，manifest: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
