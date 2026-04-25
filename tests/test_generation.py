from lib.generation import (
    apply_temperature_workaround,
    build_layers_config,
    format_status,
)


def test_temperature_workaround_bumps_one():
    assert apply_temperature_workaround(1.0) == 1.001


def test_temperature_workaround_passthrough():
    assert apply_temperature_workaround(0.5) == 0.5
    assert apply_temperature_workaround(1.5) == 1.5


def test_build_layers_config_length_matches_n_vq():
    params = {
        "n_vq": 8,
        "text_temperature": 1.5,
        "text_top_p": 1.0,
        "text_top_k": 50,
        "audio_temperature": 0.95,
        "audio_top_p": 0.95,
        "audio_top_k": 50,
        "audio_repetition_penalty": 1.1,
    }
    layers = build_layers_config(params)
    assert len(layers) == 1 + 8  # 1 text layer + n_vq audio layers


def test_build_layers_config_text_layer_has_no_rep_penalty():
    params = {
        "n_vq": 4,
        "text_temperature": 1.5,
        "text_top_p": 1.0,
        "text_top_k": 50,
        "audio_temperature": 0.95,
        "audio_top_p": 0.95,
        "audio_top_k": 50,
        "audio_repetition_penalty": 1.1,
    }
    layers = build_layers_config(params)
    assert layers[0]["temperature"] == 1.5
    assert layers[0]["repetition_penalty"] == 1.0
    assert layers[1]["temperature"] == 0.95
    assert layers[1]["repetition_penalty"] == 1.1


def test_format_status_contains_key_metrics():
    status = format_status(
        text_length=100,
        max_tokens=2500,
        gen_time_s=12.0,
        audio_duration_s=20.0,
        vram_gb=4.5,
        n_vq=16,
        voice_source="audio",
    )
    assert "100" in status
    assert "16" in status
    assert "20.0" in status or "20.00" in status
    assert "RTF" in status
