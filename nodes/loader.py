"""ComfyUI node: MOSSTTSLoader."""

import time

from ..lib.model_state import get_or_load


class MOSSTTSLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (
                    "STRING",
                    {"default": "OpenMOSS-Team/MOSS-TTS-Local-Transformer", "multiline": False},
                ),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["auto", "fp16", "bf16", "fp32"], {"default": "fp16"}),
                "attn_implementation": (
                    ["auto", "sdpa", "flash_attention_2", "eager"],
                    {"default": "sdpa"},
                ),
                "keep_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MOSS_TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MOSS-TTS/loaders"

    @classmethod
    def IS_CHANGED(cls, model_id, device, dtype, attn_implementation, keep_loaded):
        # When keep_loaded is False, the previous run's Generate node freed the
        # model from the cache. ComfyUI's per-node output cache would otherwise
        # hand the now-empty handle to Generate again. Returning a fresh value
        # each call forces ComfyUI to re-execute load() and reload the model.
        if not keep_loaded:
            return time.time()
        return f"{model_id}|{device}|{dtype}|{attn_implementation}|keep"

    def load(self, model_id, device, dtype, attn_implementation, keep_loaded):
        entry = get_or_load(
            model_id=model_id,
            device=device,
            dtype_str=dtype,
            attn_impl=attn_implementation,
            keep_loaded=keep_loaded,
        )
        return (entry,)
