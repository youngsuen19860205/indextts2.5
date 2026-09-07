"""Discovery and planning of batch voice cloning tasks (no torch required)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_UNSAFE_CHARS = re.compile(r"[^\w.\-]+", re.UNICODE)
_MAX_NAME_LENGTH = 100


class BatchInputError(ValueError):
    """Raised when the reference / gen_text inputs are invalid."""


@dataclass(frozen=True)
class ReferenceSample:
    """A reference speaker: a WAV file plus its same-basename transcript."""

    name: str
    wav_path: Path
    text_path: Path
    text: str


@dataclass(frozen=True)
class TextTask:
    """A single ``gen_text/*.txt`` file to synthesize."""

    name: str
    text_path: Path
    text: str


@dataclass(frozen=True)
class SynthesisTask:
    """One reference x one text, plus the resolved output path."""

    reference: ReferenceSample
    text_task: TextTask
    output_path: Path


def sanitize_name(name: str) -> str:
    """Return a filesystem-safe version of *name*.

    Unicode letters (e.g. Chinese) are preserved, path separators and other
    unsafe characters are replaced by ``_``.
    """
    normalized = unicodedata.normalize("NFKC", name).strip()
    safe = _UNSAFE_CHARS.sub("_", normalized).strip("._")
    if not safe:
        safe = "unnamed"
    return safe[:_MAX_NAME_LENGTH]


def read_text_file(path: Path) -> str:
    """Read a UTF-8 text file and return its stripped content."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise BatchInputError(f"文本文件不是合法的 UTF-8 编码: {path}") from exc
    except OSError as exc:
        raise BatchInputError(f"无法读取文本文件 {path}: {exc}") from exc
    return content.strip()


def discover_references(
    reference_dir: Path, selected: list[str] | None = None
) -> list[ReferenceSample]:
    """Pair ``<name>.wav`` with ``<name>.txt`` inside *reference_dir*.

    Args:
        reference_dir: directory holding the reference samples.
        selected: optional list of basenames to keep. When omitted, every
            valid pair is used (sorted by name, deterministic).
    """
    if not reference_dir.is_dir():
        raise BatchInputError(f"reference 目录不存在: {reference_dir}")

    wav_paths = sorted(
        (p for p in reference_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"),
        key=lambda p: p.name,
    )
    if not wav_paths:
        raise BatchInputError(f"reference 目录中没有 .wav 文件: {reference_dir}")

    samples: list[ReferenceSample] = []
    missing_text: list[str] = []
    empty_text: list[str] = []
    for wav_path in wav_paths:
        text_path = wav_path.with_suffix(".txt")
        if not text_path.is_file():
            missing_text.append(wav_path.name)
            continue
        text = read_text_file(text_path)
        if not text:
            empty_text.append(text_path.name)
            continue
        samples.append(
            ReferenceSample(
                name=wav_path.stem,
                wav_path=wav_path,
                text_path=text_path,
                text=text,
            )
        )

    if missing_text:
        raise BatchInputError("以下参考音频缺少同名 .txt 参考文本: " + ", ".join(missing_text))
    if empty_text:
        raise BatchInputError("以下参考文本为空: " + ", ".join(empty_text))
    if not samples:
        raise BatchInputError(f"reference 目录中没有有效的 wav/txt 配对: {reference_dir}")

    if selected is not None:
        by_name = {sample.name: sample for sample in samples}
        unknown = [name for name in selected if name not in by_name]
        if unknown:
            available = ", ".join(sorted(by_name))
            raise BatchInputError(
                f"未找到指定的 reference: {', '.join(unknown)}（可用: {available}）"
            )
        samples = [by_name[name] for name in selected]

    return samples


def discover_texts(gen_text_dir: Path) -> list[TextTask]:
    """Collect every non-empty ``gen_text/*.txt`` file, sorted by filename."""
    if not gen_text_dir.is_dir():
        raise BatchInputError(f"gen_text 目录不存在: {gen_text_dir}")

    text_paths = sorted(
        (p for p in gen_text_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"),
        key=lambda p: p.name,
    )
    if not text_paths:
        raise BatchInputError(f"gen_text 目录中没有 .txt 文件: {gen_text_dir}")

    tasks: list[TextTask] = []
    empty: list[str] = []
    for text_path in text_paths:
        text = read_text_file(text_path)
        if not text:
            empty.append(text_path.name)
            continue
        tasks.append(TextTask(name=text_path.stem, text_path=text_path, text=text))

    if empty:
        raise BatchInputError("以下待合成文本为空: " + ", ".join(empty))
    return tasks


#: Languages accepted by ``IndexTTS2.infer(..., lang=...)`` in IndexTTS 2.5.
SUPPORTED_LANGUAGES = ("zh", "en", "zhen", "ja", "es", "ar")


def resolve_language(text_name: str, default_language: str) -> str:
    """Return the language for a ``gen_text`` file.

    A ``<name>.<lang>.txt`` suffix (e.g. ``greeting.en.txt``) overrides
    *default_language*, which makes cross-lingual batches easy to express.
    """
    _, _, suffix = text_name.rpartition(".")
    if suffix and suffix.lower() in SUPPORTED_LANGUAGES:
        return suffix.lower()
    return default_language.lower()


def build_output_name(
    text_name: str, reference_name: str, template: str = "{text}__{reference}"
) -> str:
    """Render *template* into a safe output filename (without extension)."""
    try:
        rendered = template.format(
            text=sanitize_name(text_name), reference=sanitize_name(reference_name)
        )
    except (KeyError, IndexError) as exc:
        raise BatchInputError(
            f"输出命名模板 {template!r} 无效，仅支持 {{text}} 和 {{reference}} 占位符"
        ) from exc
    return sanitize_name(rendered)


def plan_tasks(
    references: list[ReferenceSample],
    texts: list[TextTask],
    gen_wav_dir: Path,
    name_template: str = "{text}__{reference}",
) -> list[SynthesisTask]:
    """Build the deterministic reference x text task list.

    Output paths are checked for collisions so that several references never
    overwrite each other's results.
    """
    tasks: list[SynthesisTask] = []
    seen: dict[Path, SynthesisTask] = {}
    for text_task in texts:
        for reference in references:
            name = build_output_name(text_task.name, reference.name, name_template)
            output_path = gen_wav_dir / f"{name}.wav"
            previous = seen.get(output_path)
            if previous is not None:
                raise BatchInputError(
                    f"输出文件名冲突: {output_path.name} 同时由 "
                    f"({previous.text_task.name}, {previous.reference.name}) 和 "
                    f"({text_task.name}, {reference.name}) 生成，"
                    "请调整 --name-template 或输入文件名"
                )
            task = SynthesisTask(reference=reference, text_task=text_task, output_path=output_path)
            seen[output_path] = task
            tasks.append(task)
    return tasks
