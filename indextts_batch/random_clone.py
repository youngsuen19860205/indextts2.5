"""Randomly pick a speaker / reference / target text and clone one utterance.

Directory layout expected by this module::

    speaker_wav_txt/
        <speaker>/
            a.wav
            a.txt          # reference transcript for a.wav
            ...
    gen_txt/
        greeting.txt       # target text to synthesize
        ...
    gen_wav/                # output directory (created if missing)

For every requested sample this module:

1. Randomly selects a speaker sub-directory under ``speaker_wav_txt/``.
2. Randomly selects one *valid* ``<name>.wav`` + ``<name>.txt`` pair from that
   speaker (entries missing either file, or with an empty transcript, are
   skipped with a warning).
3. Randomly selects one non-empty ``gen_txt/*.txt`` as the text to synthesize.
4. Calls the existing IndexTTS 2.5 inference pipeline (reusing
   ``indextts_batch.cli`` model loading and ``indextts_batch.audio`` output
   conversion -- no model loading logic is duplicated here).
5. Writes ``<reference_wav_stem>_gen_wav.wav`` to ``gen_wav/`` plus a sibling
   ``<reference_wav_stem>_gen_wav.txt`` description file. If either file
   already exists, an incrementing ``_2``, ``_3``, ... suffix is appended so
   that nothing is silently overwritten and the wav/txt pair always shares
   the same stem.

Example::

    python -m indextts_batch.random_clone --count 3 --seed 42 \\
        --speaker-wav-txt-dir speaker_wav_txt --gen-txt-dir gen_txt \\
        --gen-wav-dir gen_wav --model-dir checkpoints --lang zh
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from indextts_batch.audio import convert_to_output_format
from indextts_batch.cli import (
    SUPPORTED_LANGUAGES,
    _load_model,
    _resolve_device,
    _set_seed,
    _validate_model_dir,
)
from indextts_batch.tasks import BatchInputError, read_text_file, resolve_language


@dataclass(frozen=True)
class SpeakerReference:
    """A single valid ``<name>.wav`` + ``<name>.txt`` pair for one speaker."""

    speaker: str
    name: str
    wav_path: Path
    text_path: Path
    text: str


@dataclass(frozen=True)
class TargetText:
    """One ``gen_txt/*.txt`` candidate target text."""

    name: str
    text_path: Path
    text: str


@dataclass(frozen=True)
class RandomPick:
    """One resolved (speaker, reference pair, target text) selection."""

    reference: SpeakerReference
    target: TargetText


def discover_speaker_dirs(speaker_root: Path) -> list[Path]:
    """Return the sorted list of speaker sub-directories under *speaker_root*."""
    if not speaker_root.is_dir():
        raise BatchInputError(f"speaker_wav_txt 目录不存在: {speaker_root}")
    speakers = sorted((p for p in speaker_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not speakers:
        raise BatchInputError(f"speaker_wav_txt 目录下没有 speaker 子目录: {speaker_root}")
    return speakers


def discover_speaker_references(speaker_dir: Path) -> list[SpeakerReference]:
    """Return every valid wav/txt pair inside one speaker directory.

    Entries whose ``.wav`` has no same-basename ``.txt`` (or vice versa), or
    whose transcript is empty, are skipped with a warning printed to stderr;
    they never abort discovery for the rest of the speaker's pairs.
    """
    speaker = speaker_dir.name
    wav_paths = sorted(
        (p for p in speaker_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"),
        key=lambda p: p.name,
    )
    pairs: list[SpeakerReference] = []
    for wav_path in wav_paths:
        text_path = wav_path.with_suffix(".txt")
        if not text_path.is_file():
            print(f">> 警告: {wav_path} 缺少同名参考文本，已跳过", file=sys.stderr)
            continue
        text = read_text_file(text_path)
        if not text:
            print(f">> 警告: {text_path} 参考文本为空，已跳过", file=sys.stderr)
            continue
        pairs.append(
            SpeakerReference(
                speaker=speaker,
                name=wav_path.stem,
                wav_path=wav_path,
                text_path=text_path,
                text=text,
            )
        )
    return pairs


def discover_all_speaker_references(
    speaker_root: Path,
) -> dict[str, list[SpeakerReference]]:
    """Map every speaker with at least one valid pair to its pair list."""
    speakers = discover_speaker_dirs(speaker_root)
    by_speaker: dict[str, list[SpeakerReference]] = {}
    for speaker_dir in speakers:
        pairs = discover_speaker_references(speaker_dir)
        if pairs:
            by_speaker[speaker_dir.name] = pairs
        else:
            print(
                f">> 警告: speaker {speaker_dir.name} 下没有有效的 wav/txt 配对，已跳过",
                file=sys.stderr,
            )
    if not by_speaker:
        raise BatchInputError(
            f"speaker_wav_txt 目录下没有任何 speaker 存在有效的 wav/txt 配对: {speaker_root}"
        )
    return by_speaker


def discover_target_texts(gen_txt_dir: Path) -> list[TargetText]:
    """Collect every non-empty ``gen_txt/*.txt``, sorted by filename."""
    if not gen_txt_dir.is_dir():
        raise BatchInputError(f"gen_txt 目录不存在: {gen_txt_dir}")

    text_paths = sorted(
        (p for p in gen_txt_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"),
        key=lambda p: p.name,
    )
    if not text_paths:
        raise BatchInputError(f"gen_txt 目录中没有 .txt 文件: {gen_txt_dir}")

    targets: list[TargetText] = []
    for text_path in text_paths:
        text = read_text_file(text_path)
        if not text:
            print(f">> 警告: {text_path} 目标文本为空，已跳过", file=sys.stderr)
            continue
        targets.append(TargetText(name=text_path.stem, text_path=text_path, text=text))

    if not targets:
        raise BatchInputError(f"gen_txt 目录中没有非空的目标文本: {gen_txt_dir}")
    return targets


def pick_random_sample(speaker_root: Path, gen_txt_dir: Path, rng: random.Random) -> RandomPick:
    """Randomly select one (speaker, reference pair, target text) combination."""
    by_speaker = discover_all_speaker_references(speaker_root)
    speaker = rng.choice(sorted(by_speaker))
    reference = rng.choice(by_speaker[speaker])
    targets = discover_target_texts(gen_txt_dir)
    target = rng.choice(targets)
    return RandomPick(reference=reference, target=target)


def resolve_output_stem(gen_wav_dir: Path, reference_name: str) -> str:
    """Return a collision-free ``<reference_name>_gen_wav[_N]`` stem.

    Both the ``.wav`` and the sibling ``.txt`` description file are checked so
    that a pre-existing description file alone cannot cause a silent
    mismatch between the two.
    """
    base = f"{reference_name}_gen_wav"
    candidate = base
    suffix = 2
    while (gen_wav_dir / f"{candidate}.wav").exists() or (
        gen_wav_dir / f"{candidate}.txt"
    ).exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def build_description(
    pick: RandomPick,
    language: str,
    output_wav_path: Path,
) -> str:
    """Render a human-readable description of one generated sample."""
    reference = pick.reference
    target = pick.target
    lines = [
        f"speaker: {reference.speaker}",
        f"reference_wav: {reference.wav_path}",
        f"reference_text_file: {reference.text_path}",
        f"reference_text: {reference.text}",
        f"target_text_file: {target.text_path}",
        f"target_text: {target.text}",
        f"language: {language}",
        f"output_wav: {output_wav_path}",
        f"generated_at: {datetime.datetime.now().isoformat(timespec='seconds')}",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m indextts_batch.random_clone",
        description=(
            "从 speaker_wav_txt/<speaker>/ 中随机选择参考音频与文本，"
            "从 gen_txt/ 中随机选择目标文本，使用 IndexTTS 2.5 生成克隆语音到 gen_wav/。"
        ),
    )
    parser.add_argument(
        "--speaker-wav-txt-dir",
        default="speaker_wav_txt",
        help="包含 <speaker>/<name>.wav + <name>.txt 的根目录（默认 speaker_wav_txt）",
    )
    parser.add_argument(
        "--gen-txt-dir", default="gen_txt", help="待合成目标文本目录（默认 gen_txt）"
    )
    parser.add_argument(
        "--gen-wav-dir", default="gen_wav", help="生成音频/描述文件输出目录（默认 gen_wav）"
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=1,
        metavar="N",
        help="生成样本数量（默认 1，每次独立随机选择，允许重复）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机选择用的种子；不设置时每次运行选择不同，设置后可复现同样的选择",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("INDEXTTS_MODEL_DIR", "checkpoints"),
        help="IndexTTS 2.5 权重目录（默认 checkpoints，可用环境变量 INDEXTTS_MODEL_DIR 覆盖）",
    )
    parser.add_argument(
        "--cfg-path", default=None, help="config.yaml 路径（默认 <model-dir>/config.yaml）"
    )
    parser.add_argument(
        "--lang",
        default="zh",
        choices=SUPPORTED_LANGUAGES,
        help="默认合成语种；目标文本 <name>.<lang>.txt 可覆盖该值",
    )
    parser.add_argument(
        "--device", default="auto", help="推理设备：auto / cuda / cuda:0 / cpu（默认 auto）"
    )
    parser.add_argument("--fp32", action="store_true", help="关闭 bf16 推理")
    parser.add_argument(
        "--emo-alpha", type=float, default=1.0, help="情感强度 emo_alpha（默认 1.0）"
    )
    parser.add_argument(
        "--interval-silence", type=int, default=200, help="分段之间插入的静音毫秒数（默认 200）"
    )
    parser.add_argument(
        "--max-text-tokens-per-segment", type=int, default=120, help="每段最大 token 数（默认 120）"
    )
    parser.add_argument(
        "--duration-factor", type=float, default=1.0, help="时长缩放因子（默认 1.0）"
    )
    parser.add_argument("--no-text-normalization", action="store_true", help="关闭上游文本正则化")
    parser.add_argument(
        "--use-random", action="store_true", help="启用上游 infer(use_random=True) 的随机采样"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true", help="单条失败时继续处理其余样本"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印随机选择结果，不加载模型也不生成音频"
    )
    parser.add_argument("--verbose", action="store_true", help="打印上游推理详细日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    speaker_root = Path(args.speaker_wav_txt_dir)
    gen_txt_dir = Path(args.gen_txt_dir)
    gen_wav_dir = Path(args.gen_wav_dir)
    model_dir = Path(args.model_dir)
    cfg_path = Path(args.cfg_path) if args.cfg_path else model_dir / "config.yaml"

    if args.count < 1:
        print("错误: --count 必须大于等于 1", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)

    try:
        picks = [pick_random_sample(speaker_root, gen_txt_dir, rng) for _ in range(args.count)]
    except BatchInputError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for index, pick in enumerate(picks, start=1):
            language = resolve_language(pick.target.name, args.lang)
            print(
                f"[{index}/{len(picks)}] speaker={pick.reference.speaker} "
                f"reference={pick.reference.wav_path.name} "
                f"target={pick.target.text_path.name} lang={language}"
            )
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

    tts = _load_model(args, device)

    failures = 0
    for index, pick in enumerate(picks, start=1):
        language = resolve_language(pick.target.name, args.lang)
        stem = resolve_output_stem(gen_wav_dir, pick.reference.name)
        output_wav_path = gen_wav_dir / f"{stem}.wav"
        output_txt_path = gen_wav_dir / f"{stem}.txt"
        raw_path = raw_dir / f"{stem}.raw.wav"

        started = time.time()
        try:
            _set_seed(args.seed if args.seed is not None else 0)
            tts.infer(
                spk_audio_prompt=str(pick.reference.wav_path),
                text=pick.target.text,
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
            convert_to_output_format(raw_path, output_wav_path)
            output_txt_path.write_text(
                build_description(pick, language, output_wav_path), encoding="utf-8"
            )
            elapsed = time.time() - started
            print(
                f"[{index}/{len(picks)}] 生成成功 speaker={pick.reference.speaker} "
                f"reference={pick.reference.wav_path.name} -> {output_wav_path} "
                f"({elapsed:.2f}s)"
            )
        except Exception as exc:  # noqa: BLE001 - report and optionally continue
            failures += 1
            print(
                f"[{index}/{len(picks)}] 生成失败 speaker={pick.reference.speaker} "
                f"reference={pick.reference.wav_path.name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                shutil.rmtree(raw_dir, ignore_errors=True)
                return 1
        finally:
            raw_path.unlink(missing_ok=True)

    shutil.rmtree(raw_dir, ignore_errors=True)
    ok = len(picks) - failures
    print(f">> 完成: 成功 {ok} / 共 {len(picks)}，失败 {failures}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
