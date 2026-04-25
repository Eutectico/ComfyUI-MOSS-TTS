"""Conversions between ComfyUI's AUDIO dict and MOSS-TTS' tensor format."""

import os
import uuid
import torch
import torchaudio


def audio_dict_to_wav(audio_dict: dict, dest_dir: str) -> str:
    """Write a ComfyUI AUDIO dict to a WAV file, return the path.

    ComfyUI AUDIO is {"waveform": [B, C, T] tensor, "sample_rate": int}.
    We write batch index 0 (first sample in batch).
    """
    os.makedirs(dest_dir, exist_ok=True)
    waveform = audio_dict["waveform"]  # [B, C, T]
    sample_rate = int(audio_dict["sample_rate"])
    if waveform.dim() != 3:
        raise ValueError(
            f"Expected ComfyUI AUDIO waveform shape [B, C, T], got {tuple(waveform.shape)}"
        )
    audio_2d = waveform[0].detach().cpu().to(torch.float32)  # [C, T]
    path = os.path.join(dest_dir, f"moss_tts_voice_{uuid.uuid4().hex}.wav")
    torchaudio.save(path, audio_2d, sample_rate)
    return path


def tts_output_to_audio_dict(audio: torch.Tensor, sample_rate: int) -> dict:
    """Pack a 1D MOSS-TTS audio tensor into ComfyUI AUDIO format [1, 1, T]."""
    if audio.dim() != 1:
        raise ValueError(f"Expected 1D audio tensor, got shape {tuple(audio.shape)}")
    return {
        "waveform": audio.unsqueeze(0).unsqueeze(0),
        "sample_rate": int(sample_rate),
    }
