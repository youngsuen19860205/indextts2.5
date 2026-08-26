"""Precision selection helpers shared by inference entry points and the WebUI."""


def resolve_gpt_precision(*, use_fp16=False, use_bf16=False, device=None):
    """Return the requested GPT precision, after applying device constraints."""
    if use_fp16 and use_bf16:
        raise ValueError("use_fp16 and use_bf16 are mutually exclusive")

    device_type = str(device).split(":", 1)[0] if device is not None else None
    if device_type in {"cpu", "mps"}:
        return None
    if use_fp16:
        return "fp16"
    if use_bf16:
        return "bf16"
    return None


def select_half_precision(*, enabled, cuda_available, native_bf16_supported):
    """Prefer native BF16 for half precision, otherwise fall back to FP16."""
    if not enabled or not cuda_available:
        return None
    return "bf16" if native_bf16_supported else "fp16"


def cuda_supports_native_bf16(torch_module):
    """Return whether CUDA provides native BF16 rather than software emulation."""
    if not torch_module.cuda.is_available():
        return False
    return torch_module.cuda.is_bf16_supported(including_emulation=False)
