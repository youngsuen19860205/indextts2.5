"""IndexTTS 2.5 batch voice cloning helpers.

This package sits on top of the official IndexTTS 2.5 inference API
(``indextts.infer_v2_5.IndexTTS2``) and adds a batch pipeline that reads
reference speakers from ``reference/``, texts from ``gen_text/`` and writes
mono / 16 kHz / 16-bit PCM WAV files to ``gen_wav/``.
"""

OUTPUT_SAMPLE_RATE = 16000
OUTPUT_CHANNELS = 1
OUTPUT_SAMPLE_WIDTH = 2

__all__ = ["OUTPUT_SAMPLE_RATE", "OUTPUT_CHANNELS", "OUTPUT_SAMPLE_WIDTH"]
