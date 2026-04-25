"""ComfyUI-MOSS-TTS — Custom nodes for MOSS-TTS 1.7B zero-shot voice cloning."""

from nodes import (
    MOSSTTSLoader,
    MOSSTTSVoiceReference,
    MOSSTTSGenerate,
    MOSSTTSUnload,
)

NODE_CLASS_MAPPINGS = {
    "MOSSTTSLoader": MOSSTTSLoader,
    "MOSSTTSVoiceReference": MOSSTTSVoiceReference,
    "MOSSTTSGenerate": MOSSTTSGenerate,
    "MOSSTTSUnload": MOSSTTSUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSTTSLoader": "MOSS-TTS Loader",
    "MOSSTTSVoiceReference": "MOSS-TTS Voice Reference",
    "MOSSTTSGenerate": "MOSS-TTS Generate",
    "MOSSTTSUnload": "MOSS-TTS Unload",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
