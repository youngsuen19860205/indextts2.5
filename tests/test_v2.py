"""
IndexTTS v2 tests.

Run with:
    uv run --extra test pytest tests/test_v2.py -v

CI only (no GPU):
    uv run --extra test pytest tests/test_v2.py -v -m "not gpu"
"""
import ast
import importlib
import sys
import types
from pathlib import Path

import pytest

CHECKPOINTS_DIR = Path("checkpoints")
CONFIG_PATH = CHECKPOINTS_DIR / "config.yaml"

CHECKPOINTS_25_DIR = Path("checkpoints_25")
CONFIG_25_PATH = CHECKPOINTS_25_DIR / "config.yaml"


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture(scope="module")
def tts_model():
    pytest.importorskip("torch")
    if not CONFIG_PATH.exists():
        pytest.skip(f"Checkpoints not found at {CHECKPOINTS_DIR}")

    from indextts.infer_v2 import IndexTTS2
    return IndexTTS2(
        cfg_path=str(CONFIG_PATH),
        model_dir=str(CHECKPOINTS_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
    )


@pytest.fixture(scope="module")
def prompt_wav():
    from indextts.utils.examples_downloader import ensure_test_sample_available
    return ensure_test_sample_available()


@pytest.fixture(scope="module")
def tts_model_25():
    pytest.importorskip("torch")
    if not CONFIG_25_PATH.exists():
        pytest.skip(f"Checkpoints not found at {CHECKPOINTS_25_DIR}")

    from indextts.infer_v2_5 import IndexTTS2
    return IndexTTS2(
        cfg_path=str(CONFIG_25_PATH),
        model_dir=str(CHECKPOINTS_25_DIR),
        use_bf16=True,
    )


# -- Download URL checks (no GPU) ---------------------------------------------

# Each auxiliary model: (test_id, repo_id, probe_file)
# probe_file: a small file in the repo to verify download works end-to-end.
_MODEL_PROBES = [
    ("bigvgan", "nvidia/bigvgan_v2_22khz_80band_256x", "config.json"),
    ("w2v-bert-2.0", "facebook/w2v-bert-2.0", "config.json"),
    ("campplus", "funasr/campplus", "campplus_cn_common.bin"),
    ("MaskGCT", "amphion/MaskGCT", "README.md"),
]


@pytest.mark.parametrize("name,repo_id,filename", _MODEL_PROBES, ids=[m[0] for m in _MODEL_PROBES])
def test_model_download_reachable(name, repo_id, filename, tmp_path):
    """Each auxiliary model must be downloadable via the real download path."""
    from indextts.utils.examples_downloader import _download_file
    from indextts.utils.network_detection import need_proxy

    base_url = "https://hf-mirror.com" if need_proxy() else "https://huggingface.co"
    url = f"{base_url}/{repo_id}/resolve/main/{filename}"
    dest = tmp_path / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    _download_file(url, str(dest), max_bytes=8192)
    assert dest.exists() and dest.stat().st_size > 0


def test_example_download_reachable(tmp_path, monkeypatch):
    """Example audio must be downloadable via the real download path."""
    from indextts.utils import examples_downloader

    monkeypatch.setattr(examples_downloader, "_TESTS_DIR", str(tmp_path))
    path = examples_downloader.download_test_sample(force=True)
    assert Path(path).exists() and Path(path).stat().st_size > 0


# -- Model download logic (no GPU) --------------------------------------------

def test_legacy_cache_compatibility(tmp_path, monkeypatch):
    """ensure_models_available preserves existing cache and skips re-download."""
    from indextts.utils import model_download
    from indextts.utils.model_download import ensure_models_available

    download_calls = []
    original_download = model_download._download_single_file

    def mock_download(*args, **kwargs):
        download_calls.append(args)
        return original_download(*args, **kwargs)

    monkeypatch.setattr(model_download, "_download_single_file", mock_download)

    model_dir = tmp_path / "checkpoints"
    cache_dir = model_dir / "hf_cache"
    cache_dir.mkdir(parents=True)

    w2v_dir = cache_dir / "w2v-bert-2.0"
    w2v_dir.mkdir()
    (w2v_dir / "config.json").write_text('{"test": true}')

    bigvgan_dir = cache_dir / "bigvgan"
    bigvgan_dir.mkdir()
    (bigvgan_dir / "config.json").write_text('{"test": true}')
    (bigvgan_dir / "bigvgan_generator.pt").write_bytes(b"fake")

    campplus = cache_dir / "campplus_cn_common.bin"
    campplus.write_bytes(b"fake_campplus")

    semantic = cache_dir / "semantic_codec_model.safetensors"
    semantic.write_bytes(b"fake_semantic")

    paths = ensure_models_available(str(model_dir))

    # Files preserved
    assert (w2v_dir / "config.json").exists()
    assert campplus.exists()
    assert semantic.exists()
    assert (bigvgan_dir / "bigvgan_generator.pt").exists()

    # No download triggered
    assert len(download_calls) == 0, f"Unexpected downloads: {download_calls}"


def test_modelscope_single_file_download_matches_local_path(tmp_path, monkeypatch):
    """ModelScope single-file download must produce the exact requested local_path."""
    from indextts.utils import model_download

    local_path = tmp_path / "hf_cache" / "semantic_codec_model.safetensors"
    expected_bytes = b"fake_semantic"

    def fake_model_file_download(model_id, file_path, local_dir):
        downloaded = Path(local_dir) / file_path
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(expected_bytes)
        return str(downloaded)

    fake_modelscope = types.ModuleType("modelscope")
    fake_hub = types.ModuleType("modelscope.hub")
    fake_file_download = types.ModuleType("modelscope.hub.file_download")
    fake_file_download.model_file_download = fake_model_file_download
    fake_hub.file_download = fake_file_download
    fake_modelscope.hub = fake_hub

    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", fake_hub)
    monkeypatch.setitem(sys.modules, "modelscope.hub.file_download", fake_file_download)
    monkeypatch.setattr(model_download, "_get_using_modelscope", lambda: True)

    got = model_download._download_single_file(
        repo_id="amphion/MaskGCT",
        filename="semantic_codec/model.safetensors",
        local_path=str(local_path),
    )

    assert got == str(local_path)
    assert local_path.exists()
    assert local_path.read_bytes() == expected_bytes


# -- IndexTTS 2.5 precision selection (no GPU) --------------------------------

def test_25_use_fp16_is_appended_to_constructor_signature():
    """The new option must not shift any existing positional arguments."""
    source = Path("indextts/infer_v2_5.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "IndexTTS2")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")

    original_args = [
        "self", "cfg_path", "model_dir", "use_bf16", "device", "use_cuda_kernel",
        "use_deepspeed", "use_accel", "use_torch_compile", "use_qwen_emo",
    ]
    arg_names = [arg.arg for arg in init.args.args]
    assert arg_names[:len(original_args)] == original_args
    assert arg_names[len(original_args)] == "use_fp16"

    default_offset = len(arg_names) - len(init.args.defaults)
    fp16_default = init.args.defaults[arg_names.index("use_fp16") - default_offset]
    assert isinstance(fp16_default, ast.Constant)
    assert fp16_default.value is False


@pytest.mark.parametrize(
    "use_fp16,use_bf16,device,expected",
    [
        (False, False, "cuda:0", None),
        (True, False, "cuda:0", "fp16"),
        (False, True, "cuda:0", "bf16"),
        (True, False, "xpu", "fp16"),
        (False, True, "xpu", "bf16"),
        (True, False, "cpu", None),
        (False, True, "cpu:0", None),
        (True, False, "mps", None),
        (False, True, "mps:0", None),
    ],
)
def test_25_resolve_gpt_precision(use_fp16, use_bf16, device, expected):
    from indextts.utils.precision import resolve_gpt_precision

    assert resolve_gpt_precision(
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        device=device,
    ) == expected


def test_25_rejects_fp16_and_bf16_together():
    from indextts.utils.precision import resolve_gpt_precision

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_gpt_precision(use_fp16=True, use_bf16=True, device="cuda:0")


@pytest.mark.parametrize(
    "enabled,cuda_available,native_bf16_supported,expected",
    [
        (False, False, False, None),
        (False, True, True, None),
        (True, False, False, None),
        (True, True, False, "fp16"),
        (True, True, True, "bf16"),
    ],
)
def test_25_selects_webui_half_precision(
        enabled, cuda_available, native_bf16_supported, expected
):
    from indextts.utils.precision import select_half_precision

    assert select_half_precision(
        enabled=enabled,
        cuda_available=cuda_available,
        native_bf16_supported=native_bf16_supported,
    ) == expected


def test_25_native_bf16_check_disables_emulation():
    from indextts.utils.precision import cuda_supports_native_bf16

    calls = []

    def is_bf16_supported(**kwargs):
        calls.append(kwargs)
        return True

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=is_bf16_supported,
        )
    )

    assert cuda_supports_native_bf16(torch_stub) is True
    assert calls == [{"including_emulation": False}]


def test_25_native_bf16_check_skips_non_cuda_devices():
    from indextts.utils.precision import cuda_supports_native_bf16

    def unexpected_probe(**kwargs):
        raise AssertionError("BF16 support should not be probed without CUDA")

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            is_bf16_supported=unexpected_probe,
        )
    )

    assert cuda_supports_native_bf16(torch_stub) is False


# -- Text segmentation (no GPU) -----------------------------------------------

def _splitter_stub():
    """A 2.5 instance with only what split_text_by_tokens touches.

    Bypasses __init__ so the test needs no checkpoints; the tokenizer stub counts
    one token per character, which is enough to drive segmentation.
    """
    import types

    from indextts.infer_v2_5 import IndexTTS2

    obj = IndexTTS2.__new__(IndexTTS2)
    obj.tokenizer = types.SimpleNamespace(encode=lambda text, **kwargs: list(text))
    obj.gpt = types.SimpleNamespace(
        text_pos_embedding=types.SimpleNamespace(
            emb=types.SimpleNamespace(num_embeddings=602)
        )
    )
    return obj


def _annotated_long_text():
    """Long enough to segment, with annotations spread across the boundaries."""
    filler = "各种应用层出不穷，而语音合成作为人机交互的重要一环，其自然度和实时性同样关键。"
    unit = filler + "他要<going|G OW1 . IH0 NG>去，银<行|XING2>也开着。"
    return unit * 5


def test_split_keeps_pronunciation_annotations_paired():
    """Segment boundaries must not land inside a <|SPECIAL_TOKEN_n|> pair.

    apply_pronunciation_annotations runs before segmentation and emits payloads
    that can contain a period -- one of the splitter's break characters -- so an
    unguarded splitter leaves halves with an unpaired marker.
    """
    from indextts.infer_v2_5 import apply_pronunciation_annotations

    splitter = _splitter_stub()
    text = apply_pronunciation_annotations(_annotated_long_text().lower())
    expected_pairs = text.count("<|SPECIAL_TOKEN_1|>") // 2 + text.count("<|SPECIAL_TOKEN_2|>") // 2
    assert expected_pairs == 10, f"fixture should carry 10 annotations, got {expected_pairs}"

    segments = splitter.split_text_by_tokens(text, 120, "<|zh|> ")
    assert len(segments) > 1, "fixture must be long enough to segment"

    unpaired = [
        s for s in segments
        if s.count("<|SPECIAL_TOKEN_1|>") % 2 or s.count("<|SPECIAL_TOKEN_2|>") % 2
    ]
    assert not unpaired, f"annotation split across segments: {unpaired}"

    kept = sum(
        s.count("<|SPECIAL_TOKEN_1|>") // 2 + s.count("<|SPECIAL_TOKEN_2|>") // 2
        for s in segments
    )
    assert kept == expected_pairs, f"lost annotations: {expected_pairs} -> {kept}"


def test_split_stays_within_position_embedding_capacity():
    """Every segment must fit text_pos_embedding, whose overrun is a CUDA assert."""
    splitter = _splitter_stub()
    capacity = splitter.gpt.text_pos_embedding.emb.num_embeddings
    prefix = "<|zh|> "

    text = "各种应用层出不穷，而语音合成作为人机交互的重要一环。" * 40
    segments = splitter.split_text_by_tokens(text, 120, prefix)
    assert len(segments) > 1

    for s in segments:
        used = splitter._token_len(prefix + s) + 1  # +1 for the trailing pad token
        assert used <= capacity, f"segment needs {used} positions, capacity is {capacity}"


def test_split_leaves_short_text_alone():
    splitter = _splitter_stub()
    text = "今天天气不错。"
    assert splitter.split_text_by_tokens(text, 120, "<|zh|> ") == [text]


def _load_qwen_emotion_module(module_name, monkeypatch):
    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    def register(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    torch = register("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.xpu = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.Tensor = object
    torch.no_grad = lambda: (lambda func: func)

    torch_nn = register("torch.nn")
    torch_nn_functional = register("torch.nn.functional")
    torch_nn_utils = register("torch.nn.utils")
    torch_nn_utils_rnn = register("torch.nn.utils.rnn", pad_sequence=lambda *args, **kwargs: None)
    torch.nn = torch_nn
    torch_nn.functional = torch_nn_functional
    torch_nn.utils = torch_nn_utils
    torch_nn_utils.rnn = torch_nn_utils_rnn

    register("torchaudio")
    register("librosa")
    register("safetensors")
    register("transformers", AutoTokenizer=_Dummy, SeamlessM4TFeatureExtractor=_Dummy, Wav2Vec2BertModel=_Dummy)
    register("modelscope", AutoModelForCausalLM=_Dummy)
    register("omegaconf", OmegaConf=types.SimpleNamespace(load=lambda *args, **kwargs: None))
    register("indextts.gpt.model_v2", UnifiedVoice=_Dummy)
    register("indextts.codec.maskgct_codec", build_semantic_codec=lambda *args, **kwargs: None)
    register("indextts.codec.models", EnhancedCodec=_Dummy)
    register("indextts.utils.checkpoint", load_checkpoint=lambda *args, **kwargs: None)
    register("indextts.utils.front", TextNormalizer=_Dummy, TextTokenizer=_Dummy)
    register("indextts.utils.tokenizer", get_tokenizer=lambda *args, **kwargs: None, lang_to_token={})
    register("indextts.utils.ja_g2p", JapaneseG2PProcessor=_Dummy)
    register("indextts.utils.nemo_tn", normalize_text=lambda text: text)
    register("indextts.s2mel.modules.commons", load_checkpoint2=lambda *args, **kwargs: None, MyModel=_Dummy)
    register("indextts.s2mel.modules.bigvgan", bigvgan=object())
    register("indextts.s2mel.modules.campplus.DTDNN", CAMPPlus=_Dummy)
    register("indextts.s2mel.modules.audio", mel_spectrogram=lambda *args, **kwargs: None)

    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", ["indextts.infer_v2", "indextts.infer_v2_5"])
def test_qwen_emotion_convert_accepts_label_values(module_name, monkeypatch):
    module = _load_qwen_emotion_module(module_name, monkeypatch)

    emo = module.QwenEmotion.__new__(module.QwenEmotion)
    emo.cn_key_to_en = {
        "高兴": "happy",
        "愤怒": "angry",
        "悲伤": "sad",
        "恐惧": "afraid",
        "反感": "disgusted",
        "低落": "melancholic",
        "惊讶": "surprised",
        "自然": "calm",
    }
    emo.desired_vector_order = list(emo.cn_key_to_en)
    emo.max_score = 1.2
    emo.min_score = 0.0

    assert emo.convert({"自然": "自然"}) == {
        "happy": 0.0,
        "angry": 0.0,
        "sad": 0.0,
        "afraid": 0.0,
        "disgusted": 0.0,
        "melancholic": 0.0,
        "surprised": 0.0,
        "calm": 1.0,
    }


@pytest.mark.parametrize("module_name", ["indextts.infer_v2", "indextts.infer_v2_5"])
def test_qwen_emotion_convert_accepts_label_only_payload(module_name, monkeypatch):
    module = _load_qwen_emotion_module(module_name, monkeypatch)

    emo = module.QwenEmotion.__new__(module.QwenEmotion)
    emo.cn_key_to_en = {
        "高兴": "happy",
        "愤怒": "angry",
        "悲伤": "sad",
        "恐惧": "afraid",
        "反感": "disgusted",
        "低落": "melancholic",
        "惊讶": "surprised",
        "自然": "calm",
    }
    emo.desired_vector_order = list(emo.cn_key_to_en)
    emo.max_score = 1.2
    emo.min_score = 0.0

    assert emo.convert({"emotion": "自然"})["calm"] == 1.0


@pytest.mark.parametrize("module_name", ["indextts.infer_v2", "indextts.infer_v2_5"])
def test_qwen_emotion_convert_redirects_cross_key_labels(module_name, monkeypatch):
    module = _load_qwen_emotion_module(module_name, monkeypatch)

    emo = module.QwenEmotion.__new__(module.QwenEmotion)
    emo.cn_key_to_en = {
        "高兴": "happy",
        "愤怒": "angry",
        "悲伤": "sad",
        "恐惧": "afraid",
        "反感": "disgusted",
        "低落": "melancholic",
        "惊讶": "surprised",
        "自然": "calm",
    }
    emo.desired_vector_order = list(emo.cn_key_to_en)
    emo.max_score = 1.2
    emo.min_score = 0.0

    emotion_dict = emo.convert({"高兴": "自然"})
    assert emotion_dict["happy"] == 0.0
    assert emotion_dict["calm"] == 1.0


# -- Inference (GPU required) --------------------------------------------------

INFER_TEXTS = [
    "大家好，这是一段测试语音。",
    "There is a vehicle arriving in dock number 7?",
    "Joseph Gordon-Levitt is an American actor.",
]


@pytest.mark.gpu
@pytest.mark.parametrize("text", INFER_TEXTS, ids=lambda t: t[:20])
def test_infer(tts_model, prompt_wav, text, tmp_path):
    out = tmp_path / "out.wav"
    tts_model.infer(spk_audio_prompt=prompt_wav, text=text, output_path=str(out))
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.gpu
def test_infer_with_emotion_vector(tts_model, prompt_wav, tmp_path):
    """infer() with explicit emotion vector."""
    emo_vec = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    out = tmp_path / "emo.wav"
    tts_model.infer(
        spk_audio_prompt=prompt_wav,
        text="今天天气真好，心情特别愉快！",
        output_path=str(out),
        emo_vector=emo_vec,
    )
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.gpu
def test_infer_with_emo_text(tts_model, prompt_wav, tmp_path):
    """infer() with use_emo_text auto-detection."""
    out = tmp_path / "emo_text.wav"
    tts_model.infer(
        spk_audio_prompt=prompt_wav,
        text="这件事让我非常生气！",
        output_path=str(out),
        use_emo_text=True,
    )
    assert out.exists() and out.stat().st_size > 1000


# -- Long text (GPU required) -------------------------------------------------

@pytest.mark.gpu
def test_infer_long_text(tts_model, prompt_wav, tmp_path):
    text = (
        "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗诺兰执导并编剧，"
        "莱昂纳多迪卡普里奥、玛丽昂歌迪亚、约瑟夫高登莱维特、艾利奥特佩吉、"
        "汤姆哈迪等联袂主演，2010年7月16日在美国上映。"
        "影片剧情游走于梦境与现实之间，讲述了由莱昂纳多扮演的造梦师，"
        "带领特工团队进入他人梦境，从他人的潜意识中盗取机密的故事。"
    )
    out = tmp_path / "long.wav"
    tts_model.infer(spk_audio_prompt=prompt_wav, text=text, output_path=str(out))
    assert out.exists() and out.stat().st_size > 5000


# -- v2.5 Inference (GPU required) ---------------------------------------------

INFER_25_TEXTS = [
    ("zh", "大家好，这是一段测试语音。"),
    ("en", "There is a vehicle arriving in dock number 7?"),
    ("en", "Joseph Gordon-Levitt is an American actor."),
]


@pytest.mark.gpu
@pytest.mark.parametrize("lang,text", INFER_25_TEXTS, ids=lambda p: p[1][:20])
def test_25_infer(tts_model_25, prompt_wav, lang, text, tmp_path):
    out = tmp_path / "out.wav"
    tts_model_25.infer(spk_audio_prompt=prompt_wav, text=text, lang=lang, output_path=str(out))
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.gpu
def test_25_infer_with_emotion_vector(tts_model_25, prompt_wav, tmp_path):
    emo_vec = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    out = tmp_path / "emo.wav"
    tts_model_25.infer(
        spk_audio_prompt=prompt_wav,
        text="今天天气真好，心情特别愉快！",
        lang="zh",
        output_path=str(out),
        emo_vector=emo_vec,
    )
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.gpu
def test_25_infer_long_text(tts_model_25, prompt_wav, tmp_path):
    text = (
        "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗诺兰执导并编剧，"
        "莱昂纳多迪卡普里奥、玛丽昂歌迪亚、约瑟夫高登莱维特、艾利奥特佩吉、"
        "汤姆哈迪等联袂主演，2010年7月16日在美国上映。"
        "影片剧情游走于梦境与现实之间，讲述了由莱昂纳多扮演的造梦师，"
        "带领特工团队进入他人梦境，从他人的潜意识中盗取机密的故事。"
    )
    out = tmp_path / "long.wav"
    tts_model_25.infer(spk_audio_prompt=prompt_wav, text=text, lang="zh", output_path=str(out))
    assert out.exists() and out.stat().st_size > 5000


@pytest.mark.gpu
def test_25_infer_with_g2p_annotation(tts_model_25, prompt_wav, tmp_path):
    out = tmp_path / "g2p.wav"
    tts_model_25.infer(
        spk_audio_prompt=prompt_wav,
        text="他在银<行|XING2>里<行|HANG2>走了半天。",
        lang="zh",
        output_path=str(out),
    )
    assert out.exists() and out.stat().st_size > 1000
