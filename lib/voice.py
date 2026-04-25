"""Resolve a ComfyUI AUDIO input or string path into a voice-reference dict."""

import logging
import os
from typing import Optional

from lib.audio import audio_dict_to_wav

log = logging.getLogger(__name__)


def build_voice_reference(
    audio: Optional[dict],
    audio_path: str,
    temp_dir: str,
) -> dict:
    """Return a voice-reference container.

    Priority:
      1. If `audio` is set, write it to a temp WAV and use that path.
         If `audio_path` is also set, log a warning and ignore it.
      2. Else if `audio_path` is non-empty, validate the path exists and use it.
      3. Else return source="none" / path=None (default voice).
    """
    if audio is not None:
        if audio_path:
            log.warning("Both audio and audio_path provided; ignoring audio_path=%r", audio_path)
        path = audio_dict_to_wav(audio, dest_dir=temp_dir)
        return {"path": path, "source": "audio"}

    if audio_path:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Voice reference audio not found: {audio_path}")
        return {"path": audio_path, "source": "path"}

    return {"path": None, "source": "none"}
