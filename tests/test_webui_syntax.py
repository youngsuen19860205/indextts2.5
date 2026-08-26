from pathlib import Path


def test_webui_python_source_parses():
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    source = webui_path.read_text(encoding="utf-8")

    assert compile(source, str(webui_path), "exec") is not None


def test_webui_wires_spk_condition_reuse_to_v25_only():
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    source = webui_path.read_text(encoding="utf-8")

    assert '"--reuse_spk_cond_for_emo"' in source
    assert 'cmd_args.reuse_spk_cond_for_emo and cmd_args.version != "2.5"' in source
    assert 'kwargs["reuse_spk_cond_for_emo"] = cmd_args.reuse_spk_cond_for_emo' in source
