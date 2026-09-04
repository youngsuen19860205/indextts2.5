"""Audio post-processing: force mono / 16 kHz / 16-bit PCM WAV output."""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from indextts_batch import OUTPUT_CHANNELS, OUTPUT_SAMPLE_RATE, OUTPUT_SAMPLE_WIDTH


class AudioProcessingError(RuntimeError):
    """Raised when an audio file cannot be read, converted or validated."""


@dataclass(frozen=True)
class WavInfo:
    """Basic PCM WAV properties."""

    sample_rate: int
    channels: int
    sample_width: int
    num_frames: int

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.num_frames / float(self.sample_rate)


def read_wav_info(path: Path) -> WavInfo:
    """Read PCM WAV properties using the standard library only."""
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise AudioProcessingError(f"{path} 不是未压缩的 PCM WAV 文件")
            return WavInfo(
                sample_rate=handle.getframerate(),
                channels=handle.getnchannels(),
                sample_width=handle.getsampwidth(),
                num_frames=handle.getnframes(),
            )
    except AudioProcessingError:
        raise
    except (wave.Error, OSError, EOFError) as exc:
        raise AudioProcessingError(f"无法解析 WAV 文件 {path}: {exc}") from exc


def validate_output_wav(path: Path) -> WavInfo:
    """Assert that *path* is a mono / 16 kHz / 16-bit PCM WAV file."""
    info = read_wav_info(path)
    problems = []
    if info.channels != OUTPUT_CHANNELS:
        problems.append(f"声道数为 {info.channels}（应为 {OUTPUT_CHANNELS}）")
    if info.sample_rate != OUTPUT_SAMPLE_RATE:
        problems.append(f"采样率为 {info.sample_rate} Hz（应为 {OUTPUT_SAMPLE_RATE} Hz）")
    if info.sample_width != OUTPUT_SAMPLE_WIDTH:
        problems.append(f"位深为 {info.sample_width * 8} bit（应为 16 bit）")
    if info.num_frames <= 0:
        problems.append("音频为空")
    if problems:
        raise AudioProcessingError(f"输出音频格式校验失败 {path}: " + "; ".join(problems))
    return info


def _resample(samples, source_rate: int, target_rate: int):
    """Resample a 1-D float numpy array, preferring soxr then librosa."""
    if source_rate == target_rate:
        return samples
    try:
        import soxr

        return soxr.resample(samples, source_rate, target_rate)
    except ImportError:
        pass
    try:
        import librosa

        return librosa.resample(samples, orig_sr=source_rate, target_sr=target_rate)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise AudioProcessingError("重采样需要 soxr 或 librosa，请先安装项目依赖") from exc


def _convert_with_soundfile(src: Path, dst: Path) -> None:
    import numpy as np
    import soundfile as sf

    try:
        samples, sample_rate = sf.read(str(src), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises RuntimeError/LibsndfileError
        raise AudioProcessingError(f"无法读取音频 {src}: {exc}") from exc

    if samples.size == 0:
        raise AudioProcessingError(f"音频内容为空: {src}")

    mono = samples.mean(axis=1)
    mono = _resample(mono, sample_rate, OUTPUT_SAMPLE_RATE)
    mono = np.clip(np.asarray(mono, dtype="float32"), -1.0, 1.0)

    try:
        sf.write(str(dst), mono, OUTPUT_SAMPLE_RATE, subtype="PCM_16", format="WAV")
    except Exception as exc:
        raise AudioProcessingError(f"无法写出音频 {dst}: {exc}") from exc


def _convert_with_ffmpeg(src: Path, dst: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioProcessingError("未找到 ffmpeg，无法进行音频格式回退转换")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        str(OUTPUT_CHANNELS),
        "-ar",
        str(OUTPUT_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace").strip() if exc.stderr else ""
        raise AudioProcessingError(f"ffmpeg 转换失败 {src} -> {dst}: {stderr}") from exc


def convert_to_output_format(src: Path, dst: Path) -> WavInfo:
    """Convert *src* into a mono / 16 kHz / 16-bit PCM WAV at *dst*.

    Python audio libraries are used first; ffmpeg is a fallback. The result is
    always validated before being returned.
    """
    if not src.is_file():
        raise AudioProcessingError(f"待转换的音频不存在: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        _convert_with_soundfile(src, dst)
    except (AudioProcessingError, ImportError) as soundfile_error:
        try:
            _convert_with_ffmpeg(src, dst)
        except AudioProcessingError as ffmpeg_error:
            raise AudioProcessingError(
                f"音频转换失败（Python 库: {soundfile_error}）（ffmpeg: {ffmpeg_error}）"
            ) from ffmpeg_error

    return validate_output_wav(dst)


def probe_reference_audio(path: Path) -> WavInfo:
    """Read the properties of a reference WAV file for validation/logging."""
    if not path.is_file():
        raise AudioProcessingError(f"参考音频不存在: {path}")
    try:
        return read_wav_info(path)
    except AudioProcessingError:
        try:
            import soundfile as sf

            info = sf.info(str(path))
        except Exception as exc:
            raise AudioProcessingError(f"无法解析参考音频 {path}: {exc}") from exc
        return WavInfo(
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            sample_width=0,
            num_frames=int(info.frames),
        )
