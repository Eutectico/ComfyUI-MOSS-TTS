"""Module-level cache of loaded MOSS-TTS models.

Keys are (model_id, device, dtype_str, attn_impl) tuples. Cache survives across
ComfyUI workflow executions but is cleared on process exit via atexit.
"""

import atexit
import gc
import logging
from typing import Optional

import torch
from transformers import AutoModel, AutoProcessor

log = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple, dict] = {}
_ATEXIT_REGISTERED = False


def _ensure_atexit():
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(cleanup_all)
        _ATEXIT_REGISTERED = True


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[requested]


def resolve_attn_impl(requested: str, device: str) -> str:
    if requested == "auto":
        return "sdpa" if device == "cuda" else "eager"
    if requested == "flash_attention_2":
        try:
            __import__("flash_attn")
        except ImportError:
            log.warning("flash_attention_2 requested but flash_attn not installed; falling back to sdpa")
            return "sdpa"
    return requested


def _set_cuda_backends():
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def get_or_load(
    model_id: str,
    device: str,
    dtype_str: str,
    attn_impl: str,
    keep_loaded: bool = True,
) -> dict:
    """Load or fetch the cached MOSS-TTS model + processor."""
    _ensure_atexit()
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype_str, resolved_device)
    resolved_attn = resolve_attn_impl(attn_impl, resolved_device)

    key = (model_id, resolved_device, dtype_str, resolved_attn)
    if key in _MODEL_CACHE:
        entry = _MODEL_CACHE[key]
        entry["keep_loaded"] = keep_loaded
        return entry

    log.info("Loading MOSS-TTS model %s on %s (dtype=%s, attn=%s)...",
             model_id, resolved_device, dtype_str, resolved_attn)
    _set_cuda_backends()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(processor, "audio_tokenizer") and processor.audio_tokenizer is not None:
        processor.audio_tokenizer = processor.audio_tokenizer.to(resolved_device)

    if resolved_device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation=resolved_attn,
        torch_dtype=resolved_dtype,
        low_cpu_mem_usage=True,
    ).to(resolved_device)
    model.eval()

    if resolved_device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    entry = {
        "model": model,
        "processor": processor,
        "device": resolved_device,
        "dtype": resolved_dtype,
        "attn_impl": resolved_attn,
        "model_id": model_id,
        "keep_loaded": keep_loaded,
    }
    _MODEL_CACHE[key] = entry
    return entry


def cleanup_all() -> str:
    """Drop every cached model and free GPU memory. Returns a status string."""
    if not _MODEL_CACHE:
        return "Model cache already empty."

    n = len(_MODEL_CACHE)
    for entry in list(_MODEL_CACHE.values()):
        entry["model"] = None
        entry["processor"] = None
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return f"Cleared {n} cached model(s). Free VRAM: {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB"
    return f"Cleared {n} cached model(s)."
