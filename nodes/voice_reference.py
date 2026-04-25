"""ComfyUI node: MOSSTTSVoiceReference."""

import os

from ..lib.voice import build_voice_reference


def _comfy_temp_dir() -> str:
    """Return ComfyUI's temp directory if available, else a process-local fallback."""
    try:
        import folder_paths  # provided by ComfyUI runtime
        return folder_paths.get_temp_directory()
    except Exception:
        path = os.path.join(os.getcwd(), "temp")
        os.makedirs(path, exist_ok=True)
        return path


class MOSSTTSVoiceReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "audio": ("AUDIO",),
                "audio_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MOSS_TTS_VOICE",)
    RETURN_NAMES = ("voice",)
    FUNCTION = "build"
    CATEGORY = "MOSS-TTS/voice"

    def build(self, audio=None, audio_path=""):
        voice = build_voice_reference(
            audio=audio,
            audio_path=audio_path,
            temp_dir=_comfy_temp_dir(),
        )
        return (voice,)
