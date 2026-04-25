import os
import pytest
from lib.voice import build_voice_reference


def test_no_inputs_returns_none_source(tmp_path):
    result = build_voice_reference(audio=None, audio_path="", temp_dir=str(tmp_path))
    assert result == {"path": None, "source": "none"}


def test_audio_path_only(tmp_path, sample_comfy_audio):
    # Write a fake wav so the path exists
    import torchaudio
    wav_path = tmp_path / "ref.wav"
    torchaudio.save(str(wav_path), sample_comfy_audio["waveform"][0], sample_comfy_audio["sample_rate"])
    result = build_voice_reference(audio=None, audio_path=str(wav_path), temp_dir=str(tmp_path))
    assert result["source"] == "path"
    assert result["path"] == str(wav_path)


def test_audio_path_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_voice_reference(audio=None, audio_path=str(tmp_path / "nope.wav"), temp_dir=str(tmp_path))


def test_audio_input_writes_temp_wav(tmp_path, sample_comfy_audio):
    result = build_voice_reference(audio=sample_comfy_audio, audio_path="", temp_dir=str(tmp_path))
    assert result["source"] == "audio"
    assert os.path.isfile(result["path"])
    assert result["path"].startswith(str(tmp_path))


def test_audio_wins_over_path(tmp_path, sample_comfy_audio, caplog):
    import torchaudio
    wav_path = tmp_path / "ref.wav"
    torchaudio.save(str(wav_path), sample_comfy_audio["waveform"][0], sample_comfy_audio["sample_rate"])
    result = build_voice_reference(
        audio=sample_comfy_audio,
        audio_path=str(wav_path),
        temp_dir=str(tmp_path),
    )
    assert result["source"] == "audio"
    assert result["path"] != str(wav_path)
    assert any("audio_path" in rec.message for rec in caplog.records)
