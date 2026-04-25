"""ComfyUI node: MOSSTTSLoader."""

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

    def load(self, model_id, device, dtype, attn_implementation, keep_loaded):
        entry = get_or_load(
            model_id=model_id,
            device=device,
            dtype_str=dtype,
            attn_impl=attn_implementation,
            keep_loaded=keep_loaded,
        )
        return (entry,)
