import os
import torch
import torchaudio
from lib.audio import audio_dict_to_wav, tts_output_to_audio_dict


def test_audio_dict_to_wav_writes_readable_file(tmp_path, sample_comfy_audio):
    path = audio_dict_to_wav(sample_comfy_audio, dest_dir=str(tmp_path))
    assert os.path.isfile(path)
    waveform, sr = torchaudio.load(path)
    assert sr == sample_comfy_audio["sample_rate"]
    assert waveform.shape[1] == sample_comfy_audio["waveform"].shape[2]


def test_audio_dict_to_wav_returns_unique_paths(tmp_path, sample_comfy_audio):
    p1 = audio_dict_to_wav(sample_comfy_audio, dest_dir=str(tmp_path))
    p2 = audio_dict_to_wav(sample_comfy_audio, dest_dir=str(tmp_path))
    assert p1 != p2


def test_tts_output_to_audio_dict_shape(sample_tts_output):
    audio, sr = sample_tts_output
    result = tts_output_to_audio_dict(audio, sr)
    assert result["sample_rate"] == sr
    waveform = result["waveform"]
    assert waveform.dim() == 3
    assert waveform.shape[0] == 1  # batch
    assert waveform.shape[1] == 1  # channels
    assert waveform.shape[2] == audio.shape[0]


def test_tts_output_preserves_values(sample_tts_output):
    audio, sr = sample_tts_output
    result = tts_output_to_audio_dict(audio, sr)
    assert torch.equal(result["waveform"][0, 0], audio)
