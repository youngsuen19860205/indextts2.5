"""
IndexTTS v2 tests.

Run with:
    uv run --extra test pytest tests/test_v2.py -v

CI only (no GPU):
    uv run --extra test pytest tests/test_v2.py -v -m "not gpu"
"""
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


# -- IndexTTS-2.5 reference preparation (no GPU) ------------------------------

class _TrackingTensor:
    def __init__(self, name):
        self.name = name
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self


def test_25_reference_embedding_uses_separate_reference_device():
    from indextts.infer_v2_5 import IndexTTS2

    model = IndexTTS2.__new__(IndexTTS2)
    model.reference_device = "cpu"
    model.device = "cuda:0"

    features = _TrackingTensor("features")
    attention = _TrackingTensor("attention")
    embedding = _TrackingTensor("embedding")

    class _Extractor:
        def __call__(self, audio, sampling_rate, return_tensors):
            assert sampling_rate == 16000
            assert return_tensors == "pt"
            return {"input_features": features, "attention_mask": attention}

    model.extract_features = _Extractor()

    def get_emb(input_features, attention_mask):
        assert input_features is features
        assert attention_mask is attention
        assert features.moves == ["cpu"]
        assert attention.moves == ["cpu"]
        return embedding

    model.get_emb = get_emb

    assert model._get_reference_embedding(object()) is embedding
    assert embedding.moves == ["cuda:0"]


def test_25_emotion_reference_cache_refreshes_only_when_path_changes():
    from indextts.infer_v2_5 import IndexTTS2

    model = IndexTTS2.__new__(IndexTTS2)
    model.cache_emo_cond = None
    model.cache_emo_audio_prompt = None
    encoded = []
    emptied = []

    model._load_and_cut_audio = lambda path, *args, **kwargs: (f"audio:{path}", 16000)
    model._get_reference_embedding = lambda audio: encoded.append(audio) or f"cond:{audio}"
    model._empty_cuda_cache = lambda: emptied.append(True)

    first = model._prepare_emotion_reference("emotion-a.wav")
    assert model._prepare_emotion_reference("emotion-a.wav") == first
    second = model._prepare_emotion_reference("emotion-b.wav")

    assert first == "cond:audio:emotion-a.wav"
    assert second == "cond:audio:emotion-b.wav"
    assert encoded == ["audio:emotion-a.wav", "audio:emotion-b.wav"]
    assert emptied == [True]


def test_25_speaker_reference_cache_hit_skips_preprocessing():
    from indextts.infer_v2_5 import IndexTTS2

    model = IndexTTS2.__new__(IndexTTS2)
    model.cache_spk_audio_prompt = "speaker.wav"
    model.cache_spk_cond = "spk"
    model.cache_s2mel_style = "style"
    model.cache_s2mel_prompt = "prompt"
    model.cache_mel = "mel"
    model._load_and_cut_audio = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("cache hit must not reload the speaker reference")
    )

    assert model._prepare_speaker_reference("speaker.wav") == (
        "spk",
        "style",
        "prompt",
        "mel",
    )


def test_25_speaker_reference_cache_miss_routes_devices(monkeypatch):
    import indextts.infer_v2_5 as infer_v25

    events = []

    class _Tensor:
        def __init__(self, name, device=None):
            self.name = name
            self.device = device

        def to(self, device):
            self.device = str(device)
            events.append(("to", self.name, self.device))
            return self

        def clone(self):
            events.append(("clone", self.name))
            return _Tensor(f"{self.name}-clone", self.device)

        def float(self):
            return self

        def size(self, dim):
            assert dim == 2
            return 42

        def mean(self, *args, **kwargs):
            return _Tensor(f"{self.name}-mean", self.device)

        def __sub__(self, other):
            return self

        def unsqueeze(self, dim):
            events.append(("unsqueeze", self.name, dim))
            return self

    class _Resample:
        def __init__(self, source_rate, target_rate):
            self.target_rate = target_rate
            events.append(("resample-init", source_rate, target_rate))

        def __call__(self, audio):
            events.append(("resample", self.target_rate))
            return _Tensor(f"audio-{self.target_rate}", "cpu")

    monkeypatch.setattr(infer_v25.torchaudio.transforms, "Resample", _Resample)

    def fbank(audio, **kwargs):
        assert audio.device == "cuda:0"
        events.append(("fbank", audio.device, kwargs["sample_frequency"]))
        return _Tensor("fbank", audio.device)

    monkeypatch.setattr(infer_v25.torchaudio.compliance.kaldi, "fbank", fbank)
    monkeypatch.setattr(infer_v25.torch, "LongTensor", lambda values: _Tensor("lengths"))

    model = infer_v25.IndexTTS2.__new__(infer_v25.IndexTTS2)
    model.device = "cuda:0"
    model.reference_device = "cpu"
    model.cache_spk_cond = None
    model.cache_s2mel_style = None
    model.cache_s2mel_prompt = None
    model.cache_spk_audio_prompt = None
    model.cache_mel = None
    model._empty_cuda_cache = lambda: events.append(("empty-cache",))

    loads = []
    model._load_and_cut_audio = lambda path, *args, **kwargs: (
        loads.append(path) or _Tensor("audio", "cpu"),
        22050,
    )

    embeddings = []

    def get_embedding(audio):
        embeddings.append(audio.name)
        return _Tensor("spk", "cuda:0")

    model._get_reference_embedding = get_embedding

    def mel_fn(audio):
        assert audio.device == "cuda:0"
        events.append(("mel", audio.device))
        return _Tensor("mel", audio.device)

    model.mel_fn = mel_fn

    class _CampPlus:
        def __call__(self, feat):
            assert feat.device == "cpu"
            events.append(("campplus", feat.device))
            return _Tensor("style", "cpu")

    model.campplus_model = _CampPlus()

    def length_regulator(spk_cond, ylens, **kwargs):
        assert spk_cond.device == "cuda:0"
        assert ylens.device == "cuda:0"
        events.append(("length-regulator", spk_cond.device, ylens.device))
        return [_Tensor("prompt", "cuda:0")]

    model.s2mel = types.SimpleNamespace(models={"length_regulator": length_regulator})

    spk, style, prompt, mel = model._prepare_speaker_reference("speaker.wav")

    assert loads == ["speaker.wav"]
    assert embeddings == ["audio-16000-clone"]
    assert [event for event in events if event[0] == "resample"] == [
        ("resample", 22050),
        ("resample", 16000),
    ]
    assert ("mel", "cuda:0") in events
    assert ("fbank", "cuda:0", 16000) in events
    assert ("to", "fbank", "cpu") in events
    assert ("campplus", "cpu") in events
    assert ("to", "style", "cuda:0") in events
    assert ("length-regulator", "cuda:0", "cuda:0") in events
    assert (spk.name, style.name, prompt.name, mel.name) == ("spk", "style", "prompt", "mel")


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
