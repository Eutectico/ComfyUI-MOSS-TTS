"""ComfyUI node: MOSSTTSUnload."""

from ..lib.model_state import cleanup_all


class MOSSTTSUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MOSS_TTS_MODEL",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "MOSS-TTS/utility"
    OUTPUT_NODE = True

    def unload(self, model):
        # `model` is unused on purpose — it acts as an execution-order trigger.
        status = cleanup_all()
        return (status,)
