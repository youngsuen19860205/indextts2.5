"""Unit tests for the random speaker/reference/text batch clone pipeline
(``indextts_batch.random_clone``).

These tests are CPU-only: they never download model weights and never run
real inference (the upstream ``IndexTTS2`` model is replaced by a fake),
so they run in the ``not gpu`` CI job alongside ``test_batch_clone.py``.
"""

import math
import struct
import wave
from pathlib import Path

import pytest

from indextts_batch.random_clone import (
    build_description,
    build_parser,
    discover_all_speaker_references,
    discover_speaker_dirs,
    discover_speaker_references,
    discover_target_texts,
    main,
    pick_random_sample,
    resolve_output_stem,
)
from indextts_batch.tasks import BatchInputError


def write_wav(path: Path, sample_rate=22050, channels=1, sample_width=2, seconds=0.2):
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
    speaker_root = tmp_path / "speaker_wav_txt"
    gen_txt_dir = tmp_path / "gen_txt"
    gen_wav_dir = tmp_path / "gen_wav"

    write_wav(speaker_root / "alice" / "a.wav")
    (speaker_root / "alice" / "a.txt").write_text("爱丽丝的参考文本。", encoding="utf-8")
    write_wav(speaker_root / "alice" / "b.wav")
    (speaker_root / "alice" / "b.txt").write_text("Another Alice reference.", encoding="utf-8")

    write_wav(speaker_root / "bob" / "c.wav")
    (speaker_root / "bob" / "c.txt").write_text("Bob's reference text.", encoding="utf-8")

    gen_txt_dir.mkdir(parents=True, exist_ok=True)
    (gen_txt_dir / "greet.txt").write_text("你好，世界。", encoding="utf-8")
    (gen_txt_dir / "thanks.txt").write_text("Thank you very much.", encoding="utf-8")

    return speaker_root, gen_txt_dir, gen_wav_dir


class TestSpeakerDiscovery:
    def test_discover_speaker_dirs(self, workspace):
        speaker_root, _, _ = workspace
        names = [p.name for p in discover_speaker_dirs(speaker_root)]
        assert names == ["alice", "bob"]

    def test_missing_root_is_an_error(self, tmp_path):
        with pytest.raises(BatchInputError, match="speaker_wav_txt 目录不存在"):
            discover_speaker_dirs(tmp_path / "absent")

    def test_empty_root_is_an_error(self, tmp_path):
        root = tmp_path / "speaker_wav_txt"
        root.mkdir()
        with pytest.raises(BatchInputError, match="没有 speaker 子目录"):
            discover_speaker_dirs(root)

    def test_valid_pairs_only(self, workspace):
        speaker_root, _, _ = workspace
        alice_dir = speaker_root / "alice"
        # Incomplete / invalid entries that must be skipped, not fatal.
        write_wav(alice_dir / "no_text.wav")
        (alice_dir / "empty.wav").touch()
        (alice_dir / "empty.txt").write_text("   \n", encoding="utf-8")

        pairs = discover_speaker_references(alice_dir)
        assert sorted(p.name for p in pairs) == ["a", "b"]
        assert all(p.speaker == "alice" for p in pairs)

    def test_all_speakers_valid_pairs(self, workspace):
        speaker_root, _, _ = workspace
        by_speaker = discover_all_speaker_references(speaker_root)
        assert set(by_speaker) == {"alice", "bob"}
        assert len(by_speaker["alice"]) == 2
        assert len(by_speaker["bob"]) == 1

    def test_speaker_with_no_valid_pairs_is_skipped(self, workspace):
        speaker_root, _, _ = workspace
        empty_speaker = speaker_root / "empty_speaker"
        empty_speaker.mkdir()
        by_speaker = discover_all_speaker_references(speaker_root)
        assert "empty_speaker" not in by_speaker

    def test_no_speaker_has_valid_pairs_is_an_error(self, tmp_path):
        root = tmp_path / "speaker_wav_txt"
        (root / "alice").mkdir(parents=True)
        write_wav(root / "alice" / "a.wav")  # no matching .txt
        with pytest.raises(BatchInputError, match="没有任何 speaker 存在有效的 wav/txt 配对"):
            discover_all_speaker_references(root)


class TestTargetTextDiscovery:
    def test_discover_target_texts(self, workspace):
        _, gen_txt_dir, _ = workspace
        targets = discover_target_texts(gen_txt_dir)
        assert [t.name for t in targets] == ["greet", "thanks"]

    def test_missing_dir_is_an_error(self, tmp_path):
        with pytest.raises(BatchInputError, match="gen_txt 目录不存在"):
            discover_target_texts(tmp_path / "absent")

    def test_empty_dir_is_an_error(self, tmp_path):
        gen_txt_dir = tmp_path / "gen_txt"
        gen_txt_dir.mkdir()
        with pytest.raises(BatchInputError, match="没有 .txt 文件"):
            discover_target_texts(gen_txt_dir)

    def test_all_blank_texts_is_an_error(self, tmp_path):
        gen_txt_dir = tmp_path / "gen_txt"
        gen_txt_dir.mkdir()
        (gen_txt_dir / "blank.txt").write_text("   \n", encoding="utf-8")
        with pytest.raises(BatchInputError, match="没有非空的目标文本"):
            discover_target_texts(gen_txt_dir)


class TestRandomSelection:
    def test_reproducible_with_seed(self, workspace):
        import random

        speaker_root, gen_txt_dir, _ = workspace
        pick_a = pick_random_sample(speaker_root, gen_txt_dir, random.Random(42))
        pick_b = pick_random_sample(speaker_root, gen_txt_dir, random.Random(42))
        assert pick_a.reference.speaker == pick_b.reference.speaker
        assert pick_a.reference.name == pick_b.reference.name
        assert pick_a.target.name == pick_b.target.name

    def test_different_seeds_can_diverge(self, workspace):
        import random

        speaker_root, gen_txt_dir, _ = workspace
        picks = {
            (
                pick_random_sample(speaker_root, gen_txt_dir, random.Random(seed)).reference.name,
                pick_random_sample(speaker_root, gen_txt_dir, random.Random(seed)).target.name,
            )
            for seed in range(20)
        }
        # With 3 references x 2 texts = 6 combinations, 20 seeds should hit more than one.
        assert len(picks) > 1


class TestOutputNaming:
    def test_first_output_has_no_suffix(self, tmp_path):
        gen_wav_dir = tmp_path / "gen_wav"
        gen_wav_dir.mkdir()
        assert resolve_output_stem(gen_wav_dir, "a") == "a_gen_wav"

    def test_collision_appends_incrementing_suffix(self, tmp_path):
        gen_wav_dir = tmp_path / "gen_wav"
        gen_wav_dir.mkdir()
        (gen_wav_dir / "a_gen_wav.wav").touch()
        assert resolve_output_stem(gen_wav_dir, "a") == "a_gen_wav_2"
        (gen_wav_dir / "a_gen_wav_2.wav").touch()
        assert resolve_output_stem(gen_wav_dir, "a") == "a_gen_wav_3"

    def test_existing_description_txt_alone_also_triggers_suffix(self, tmp_path):
        gen_wav_dir = tmp_path / "gen_wav"
        gen_wav_dir.mkdir()
        (gen_wav_dir / "a_gen_wav.txt").touch()
        assert resolve_output_stem(gen_wav_dir, "a") == "a_gen_wav_2"

    def test_description_contents(self, workspace):
        speaker_root, gen_txt_dir, gen_wav_dir = workspace
        import random

        pick = pick_random_sample(speaker_root, gen_txt_dir, random.Random(1))
        description = build_description(pick, "zh", gen_wav_dir / "a_gen_wav.wav")
        assert f"speaker: {pick.reference.speaker}" in description
        assert f"reference_wav: {pick.reference.wav_path}" in description
        assert f"reference_text_file: {pick.reference.text_path}" in description
        assert f"reference_text: {pick.reference.text}" in description
        assert f"target_text_file: {pick.target.text_path}" in description
        assert f"target_text: {pick.target.text}" in description
        assert "output_wav:" in description
        assert "generated_at:" in description


class TestParserDefaults:
    def test_defaults(self):
        args = build_parser().parse_args([])
        assert args.speaker_wav_txt_dir == "speaker_wav_txt"
        assert args.gen_txt_dir == "gen_txt"
        assert args.gen_wav_dir == "gen_wav"
        assert args.count == 1
        assert args.seed is None

    def test_invalid_count_is_rejected(self, workspace, capsys):
        speaker_root, gen_txt_dir, gen_wav_dir = workspace
        exit_code = main(
            [
                "--speaker-wav-txt-dir",
                str(speaker_root),
                "--gen-txt-dir",
                str(gen_txt_dir),
                "--gen-wav-dir",
                str(gen_wav_dir),
                "--count",
                "0",
            ]
        )
        assert exit_code == 2
        assert "--count 必须大于等于 1" in capsys.readouterr().err


class TestDryRun:
    def test_dry_run_lists_selection_without_model(self, workspace, capsys):
        speaker_root, gen_txt_dir, gen_wav_dir = workspace
        exit_code = main(
            [
                "--speaker-wav-txt-dir",
                str(speaker_root),
                "--gen-txt-dir",
                str(gen_txt_dir),
                "--gen-wav-dir",
                str(gen_wav_dir),
                "--count",
                "2",
                "--seed",
                "7",
                "--dry-run",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "--dry-run" in captured.out
        assert not gen_wav_dir.exists()

    def test_dry_run_reports_missing_dir(self, tmp_path, capsys):
        exit_code = main(
            [
                "--speaker-wav-txt-dir",
                str(tmp_path / "absent"),
                "--gen-txt-dir",
                str(tmp_path),
                "--gen-wav-dir",
                str(tmp_path / "gen_wav"),
                "--dry-run",
            ]
        )
        assert exit_code == 2
        assert "speaker_wav_txt 目录不存在" in capsys.readouterr().err


class FakeTTS:
    """Stands in for ``indextts.infer_v2_5.IndexTTS2`` (no torch, no weights)."""

    instances = 0

    def __init__(self):
        FakeTTS.instances += 1
        self.calls = []

    def infer(self, spk_audio_prompt, text, output_path, lang, **kwargs):
        self.calls.append((spk_audio_prompt, text, lang))
        write_wav(Path(output_path), sample_rate=22050, channels=1, seconds=0.2)
        return output_path


class TestBatchRun:
    @pytest.fixture()
    def patched(self, monkeypatch):
        pytest.importorskip("numpy")
        pytest.importorskip("soundfile")
        from indextts_batch import random_clone

        FakeTTS.instances = 0
        fake = FakeTTS()
        monkeypatch.setattr(random_clone, "_validate_model_dir", lambda *a, **k: None)
        monkeypatch.setattr(random_clone, "_resolve_device", lambda requested: "cpu")
        monkeypatch.setattr(random_clone, "_set_seed", lambda seed: None)
        monkeypatch.setattr(random_clone, "_load_model", lambda args, device: fake)
        return fake

    def test_generates_wav_and_description(self, workspace, patched):
        speaker_root, gen_txt_dir, gen_wav_dir = workspace

        exit_code = main(
            [
                "--speaker-wav-txt-dir",
                str(speaker_root),
                "--gen-txt-dir",
                str(gen_txt_dir),
                "--gen-wav-dir",
                str(gen_wav_dir),
                "--count",
                "1",
                "--seed",
                "3",
            ]
        )

        assert exit_code == 0
        assert FakeTTS.instances == 1
        wavs = sorted(gen_wav_dir.glob("*_gen_wav.wav"))
        assert len(wavs) == 1
        txt_path = wavs[0].with_suffix(".txt")
        assert txt_path.is_file()
        content = txt_path.read_text(encoding="utf-8")
        assert "speaker:" in content
        assert "reference_wav:" in content
        assert "target_text:" in content
        assert not (gen_wav_dir / "raw").exists()

    def test_repeated_run_does_not_overwrite(self, workspace, patched):
        speaker_root, gen_txt_dir, gen_wav_dir = workspace

        for _ in range(2):
            exit_code = main(
                [
                    "--speaker-wav-txt-dir",
                    str(speaker_root),
                    "--gen-txt-dir",
                    str(gen_txt_dir),
                    "--gen-wav-dir",
                    str(gen_wav_dir),
                    "--count",
                    "1",
                    "--seed",
                    "3",
                ]
            )
            assert exit_code == 0

        wavs = sorted(gen_wav_dir.glob("*_gen_wav*.wav"))
        assert len(wavs) == 2
        stems = {p.stem for p in wavs}
        assert any(stem.endswith("_2") for stem in stems)
        for wav in wavs:
            assert wav.with_suffix(".txt").is_file()

    def test_inference_failure_is_reported(self, workspace, patched, monkeypatch):
        speaker_root, gen_txt_dir, gen_wav_dir = workspace

        def failing_infer(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(patched, "infer", failing_infer)

        exit_code = main(
            [
                "--speaker-wav-txt-dir",
                str(speaker_root),
                "--gen-txt-dir",
                str(gen_txt_dir),
                "--gen-wav-dir",
                str(gen_wav_dir),
                "--count",
                "1",
                "--seed",
                "3",
            ]
        )
        assert exit_code == 1
        assert not list(gen_wav_dir.glob("*.wav"))
