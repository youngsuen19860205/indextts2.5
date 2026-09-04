"""Unit tests for the IndexTTS 2.5 batch voice cloning pipeline.

These tests are CPU-only: they never download model weights and never run real
inference, so they run in the `not gpu` CI job.

Run with:
    uv run --extra test pytest tests/test_batch_clone.py -v
"""

import csv
import json
import math
import struct
import wave
from pathlib import Path

import pytest

from indextts_batch.audio import (
    AudioProcessingError,
    convert_to_output_format,
    read_wav_info,
    validate_output_wav,
)
from indextts_batch.cli import build_parser, main
from indextts_batch.tasks import (
    BatchInputError,
    build_output_name,
    discover_references,
    discover_texts,
    plan_tasks,
    resolve_language,
    sanitize_name,
)


def write_wav(path: Path, sample_rate=22050, channels=1, sample_width=2, seconds=0.25):
    """Write a small sine-wave PCM WAV file."""
    num_frames = int(sample_rate * seconds)
    max_value = 2 ** (8 * sample_width - 1) - 1
    frames = bytearray()
    for index in range(num_frames):
        value = int(0.5 * max_value * math.sin(2 * math.pi * 440 * index / sample_rate))
        for _ in range(channels):
            frames += struct.pack("<h" if sample_width == 2 else "<i", value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))
    return path


@pytest.fixture()
def workspace(tmp_path):
    reference_dir = tmp_path / "reference"
    gen_text_dir = tmp_path / "gen_text"
    gen_wav_dir = tmp_path / "gen_wav"
    write_wav(reference_dir / "speaker_zh.wav")
    (reference_dir / "speaker_zh.txt").write_text("这是参考文本。", encoding="utf-8")
    gen_text_dir.mkdir(parents=True, exist_ok=True)
    (gen_text_dir / "hello.txt").write_text("你好，世界。", encoding="utf-8")
    return reference_dir, gen_text_dir, gen_wav_dir


class TestReferencePairing:
    def test_pairs_wav_and_txt(self, workspace):
        reference_dir, _, _ = workspace
        samples = discover_references(reference_dir)
        assert [sample.name for sample in samples] == ["speaker_zh"]
        assert samples[0].text == "这是参考文本。"
        assert samples[0].wav_path.name == "speaker_zh.wav"

    def test_missing_transcript_is_an_error(self, workspace):
        reference_dir, _, _ = workspace
        write_wav(reference_dir / "speaker_en.wav")
        with pytest.raises(BatchInputError, match="speaker_en.wav"):
            discover_references(reference_dir)

    def test_empty_transcript_is_an_error(self, workspace):
        reference_dir, _, _ = workspace
        write_wav(reference_dir / "speaker_en.wav")
        (reference_dir / "speaker_en.txt").write_text("   \n", encoding="utf-8")
        with pytest.raises(BatchInputError, match="参考文本为空"):
            discover_references(reference_dir)

    def test_explicit_selection(self, workspace):
        reference_dir, _, _ = workspace
        write_wav(reference_dir / "speaker_en.wav")
        (reference_dir / "speaker_en.txt").write_text("Reference text.", encoding="utf-8")
        samples = discover_references(reference_dir, ["speaker_en"])
        assert [sample.name for sample in samples] == ["speaker_en"]

    def test_unknown_selection_is_an_error(self, workspace):
        reference_dir, _, _ = workspace
        with pytest.raises(BatchInputError, match="未找到指定的 reference"):
            discover_references(reference_dir, ["nope"])

    def test_missing_directory_is_an_error(self, tmp_path):
        with pytest.raises(BatchInputError, match="reference 目录不存在"):
            discover_references(tmp_path / "absent")


class TestTextDiscovery:
    def test_sorted_discovery(self, workspace):
        _, gen_text_dir, _ = workspace
        (gen_text_dir / "abc.txt").write_text("Hello world.", encoding="utf-8")
        tasks = discover_texts(gen_text_dir)
        assert [task.name for task in tasks] == ["abc", "hello"]
        assert tasks[0].text == "Hello world."

    def test_empty_text_is_an_error(self, workspace):
        _, gen_text_dir, _ = workspace
        (gen_text_dir / "blank.txt").write_text("\n\n", encoding="utf-8")
        with pytest.raises(BatchInputError, match="待合成文本为空"):
            discover_texts(gen_text_dir)

    def test_language_suffix_overrides_default(self):
        assert resolve_language("greeting.en", "zh") == "en"
        assert resolve_language("greeting", "zh") == "zh"
        assert resolve_language("greeting.unknown", "ja") == "ja"


class TestOutputNaming:
    def test_sanitize_name(self):
        assert sanitize_name("a/b c") == "a_b_c"
        assert sanitize_name("  中文 语音  ") == "中文_语音"
        assert sanitize_name("../../etc/passwd") == "etc_passwd"
        assert sanitize_name("...") == "unnamed"

    def test_default_template_includes_reference(self):
        assert build_output_name("hello", "speaker_zh") == "hello__speaker_zh"

    def test_custom_template(self):
        assert build_output_name("hello", "spk", "{reference}-{text}") == "spk-hello"

    def test_invalid_template(self):
        with pytest.raises(BatchInputError):
            build_output_name("hello", "spk", "{unknown}")

    def test_plan_has_no_collisions_with_multiple_references(self, workspace):
        reference_dir, gen_text_dir, gen_wav_dir = workspace
        write_wav(reference_dir / "speaker_en.wav")
        (reference_dir / "speaker_en.txt").write_text("Reference.", encoding="utf-8")
        tasks = plan_tasks(
            discover_references(reference_dir), discover_texts(gen_text_dir), gen_wav_dir
        )
        outputs = [task.output_path.name for task in tasks]
        assert sorted(outputs) == ["hello__speaker_en.wav", "hello__speaker_zh.wav"]
        assert len(set(outputs)) == len(outputs)

    def test_colliding_template_is_rejected(self, workspace):
        reference_dir, gen_text_dir, gen_wav_dir = workspace
        write_wav(reference_dir / "speaker_en.wav")
        (reference_dir / "speaker_en.txt").write_text("Reference.", encoding="utf-8")
        with pytest.raises(BatchInputError, match="输出文件名冲突"):
            plan_tasks(
                discover_references(reference_dir),
                discover_texts(gen_text_dir),
                gen_wav_dir,
                "{text}",
            )


class TestAudioPostProcessing:
    def test_validate_rejects_wrong_format(self, tmp_path):
        path = write_wav(tmp_path / "bad.wav", sample_rate=22050, channels=2)
        with pytest.raises(AudioProcessingError, match="输出音频格式校验失败"):
            validate_output_wav(path)

    def test_validate_accepts_16k_mono_pcm16(self, tmp_path):
        path = write_wav(tmp_path / "good.wav", sample_rate=16000, channels=1)
        info = validate_output_wav(path)
        assert (info.sample_rate, info.channels, info.sample_width) == (16000, 1, 2)

    def test_conversion_produces_16k_mono_pcm16(self, tmp_path):
        pytest.importorskip("numpy")
        pytest.importorskip("soundfile")
        src = write_wav(tmp_path / "src.wav", sample_rate=44100, channels=2, seconds=0.5)
        dst = tmp_path / "out" / "dst.wav"
        info = convert_to_output_format(src, dst)
        assert info.sample_rate == 16000
        assert info.channels == 1
        assert info.sample_width == 2
        assert info.duration_seconds == pytest.approx(0.5, abs=0.02)
        assert read_wav_info(dst).num_frames > 0

    def test_conversion_of_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(AudioProcessingError, match="待转换的音频不存在"):
            convert_to_output_format(tmp_path / "absent.wav", tmp_path / "out.wav")


class TestDryRun:
    def test_dry_run_lists_tasks_without_model(self, workspace, capsys):
        reference_dir, gen_text_dir, gen_wav_dir = workspace
        exit_code = main(
            [
                "--reference-dir",
                str(reference_dir),
                "--gen-text-dir",
                str(gen_text_dir),
                "--gen-wav-dir",
                str(gen_wav_dir),
                "--dry-run",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "共规划 1 个合成任务" in captured.out
        assert "hello__speaker_zh.wav" in captured.out
        assert not gen_wav_dir.exists()

    def test_dry_run_reports_input_errors(self, tmp_path, capsys):
        exit_code = main(
            [
                "--reference-dir",
                str(tmp_path / "absent"),
                "--gen-text-dir",
                str(tmp_path),
                "--gen-wav-dir",
                str(tmp_path / "gen_wav"),
                "--dry-run",
            ]
        )
        assert exit_code == 2
        assert "reference 目录不存在" in capsys.readouterr().err

    def test_parser_defaults(self):
        args = build_parser().parse_args([])
        assert args.lang == "zh"
        assert args.device == "auto"
        assert args.seed == 0
        assert args.name_template == "{text}__{reference}"


class FakeTTS:
    """Stands in for ``indextts.infer_v2_5.IndexTTS2`` (no torch, no weights)."""

    instances = 0

    def __init__(self):
        FakeTTS.instances += 1
        self.calls = []

    def infer(self, spk_audio_prompt, text, output_path, lang, **kwargs):
        self.calls.append((spk_audio_prompt, text, lang))
        # The upstream model writes 22.05 kHz audio.
        write_wav(Path(output_path), sample_rate=22050, channels=1, seconds=0.3)
        return output_path


class TestBatchRun:
    @pytest.fixture()
    def patched(self, monkeypatch):
        pytest.importorskip("numpy")
        pytest.importorskip("soundfile")
        from indextts_batch import cli

        FakeTTS.instances = 0
        fake = FakeTTS()
        monkeypatch.setattr(cli, "_validate_model_dir", lambda *args, **kwargs: None)
        monkeypatch.setattr(cli, "_resolve_device", lambda requested: "cpu")
        monkeypatch.setattr(cli, "_set_seed", lambda seed: None)
        monkeypatch.setattr(cli, "_load_model", lambda args, device: fake)
        return fake

    def test_batch_generates_validated_16k_outputs(self, workspace, patched):
        reference_dir, gen_text_dir, gen_wav_dir = workspace
        (gen_text_dir / "greeting.en.txt").write_text("Hello there.", encoding="utf-8")

        exit_code = main(
            [
                "--reference-dir",
                str(reference_dir),
                "--gen-text-dir",
                str(gen_text_dir),
                "--gen-wav-dir",
                str(gen_wav_dir),
                "--lang",
                "zh",
            ]
        )

        assert exit_code == 0
        # The model is instantiated exactly once for the whole batch.
        assert FakeTTS.instances == 1
        assert [call[2] for call in patched.calls] == ["EN", "ZH"]

        outputs = sorted(p.name for p in gen_wav_dir.glob("*.wav"))
        assert outputs == ["greeting.en__speaker_zh.wav", "hello__speaker_zh.wav"]
        for output in gen_wav_dir.glob("*.wav"):
            info = validate_output_wav(output)
            assert (info.sample_rate, info.channels, info.sample_width) == (16000, 1, 2)
        assert not (gen_wav_dir / "raw").exists()

        manifest = gen_wav_dir / "manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert {row["status"] for row in rows} == {"ok"}
        assert {row["sample_rate"] for row in rows} == {16000}
        assert rows[0]["reference_text"] == "这是参考文本。"

    def test_failure_is_reported_and_batch_can_continue(self, workspace, patched, monkeypatch):
        reference_dir, gen_text_dir, gen_wav_dir = workspace
        (gen_text_dir / "second.txt").write_text("另一条文本。", encoding="utf-8")

        original_infer = patched.infer

        def flaky_infer(spk_audio_prompt, text, output_path, lang, **kwargs):
            if "hello" in Path(output_path).name:
                raise RuntimeError("boom")
            return original_infer(spk_audio_prompt, text, output_path, lang, **kwargs)

        monkeypatch.setattr(patched, "infer", flaky_infer)

        exit_code = main(
            [
                "--reference-dir",
                str(reference_dir),
                "--gen-text-dir",
                str(gen_text_dir),
                "--gen-wav-dir",
                str(gen_wav_dir),
                "--manifest",
                str(gen_wav_dir / "manifest.csv"),
                "--continue-on-error",
            ]
        )

        assert exit_code == 1
        rows = list(
            csv.DictReader((gen_wav_dir / "manifest.csv").read_text(encoding="utf-8").splitlines())
        )
        statuses = {row["text_file"].split("/")[-1]: row["status"] for row in rows}
        assert statuses == {"hello.txt": "failed", "second.txt": "ok"}
        assert "RuntimeError: boom" in "".join(row["error"] for row in rows)
