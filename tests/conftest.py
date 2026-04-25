import pytest
import torch


@pytest.fixture
def sample_comfy_audio():
    """A ComfyUI AUDIO dict with a 1-second sine wave at 24 kHz, mono."""
    sample_rate = 24000
    t = torch.linspace(0, 1, sample_rate)
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    return {"waveform": waveform, "sample_rate": sample_rate}


@pytest.fixture
def sample_tts_output():
    """A 1D audio tensor as produced by MOSS-TTS' processor.decode."""
    sample_rate = 24000
    t = torch.linspace(0, 1, sample_rate)
    return torch.sin(2 * torch.pi * 220 * t), sample_rate
