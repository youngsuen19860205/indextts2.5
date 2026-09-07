"""Randomly select reference speakers and run the existing batch pipeline.

Examples::

    python -m indextts_batch.random_reference --random-reference-count 1 -- --lang en
    python -m indextts_batch.random_reference \
        --random-reference-count 3 --selection-seed 42 -- --dry-run

The selected references are passed to ``indextts_batch.cli.main`` in one call,
so IndexTTS is still loaded only once for the whole batch.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from indextts_batch.cli import main as batch_main
from indextts_batch.tasks import BatchInputError, ReferenceSample, discover_references


def select_random_references(
    reference_dir: Path,
    count: int,
    seed: int | None = None,
) -> list[ReferenceSample]:
    """Select *count* valid reference pairs without replacement.

    When *seed* is provided, selection is reproducible. With no seed, Python's
    system-seeded random generator produces a fresh selection for each run.
    """
    references = discover_references(reference_dir)
    if count < 1:
        raise BatchInputError("--random-reference-count 必须大于等于 1")
    if count > len(references):
        raise BatchInputError(
            f"请求随机选择 {count} 个 reference，但目录中只有 {len(references)} 个有效配对"
        )

    return random.Random(seed).sample(references, count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m indextts_batch.random_reference",
        description=(
            "从 reference/ 随机选择一个或多个参考音频，再调用 IndexTTS 2.5 批量生成。"
            "未识别的参数会继续传递给 python -m indextts_batch。"
        ),
    )
    parser.add_argument(
        "--reference-dir",
        default="reference",
        help="参考 WAV/TXT 目录（默认 reference）",
    )
    parser.add_argument(
        "--random-reference-count",
        type=int,
        default=1,
        metavar="N",
        help="每次运行随机选择的 reference 数量（默认 1，不重复抽取）",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=None,
        help="reference 抽样随机种子；不设置时每次运行重新随机，设置后可复现",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, batch_args = parser.parse_known_args(argv)

    if batch_args and batch_args[0] == "--":
        batch_args = batch_args[1:]

    if "--reference" in batch_args:
        parser.error("随机选择模式不能同时使用 --reference；请二选一")
    if "--reference-dir" in batch_args:
        parser.error("--reference-dir 请放在随机选择参数中，不要重复传递")

    try:
        selected = select_random_references(
            Path(args.reference_dir),
            args.random_reference_count,
            args.selection_seed,
        )
    except BatchInputError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    names = ", ".join(reference.name for reference in selected)
    seed_text = "系统随机" if args.selection_seed is None else str(args.selection_seed)
    print(f">> 随机选择 {len(selected)} 个 reference（selection_seed={seed_text}）: {names}")

    forwarded = ["--reference-dir", args.reference_dir]
    for reference in selected:
        forwarded.extend(["--reference", reference.name])
    forwarded.extend(batch_args)
    return batch_main(forwarded)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
