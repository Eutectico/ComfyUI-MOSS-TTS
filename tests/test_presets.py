import pytest
from lib.presets import PRESETS, PRESET_NAMES, resolve_preset

REQUIRED_KEYS = {
    "n_vq", "text_temperature", "text_top_p", "text_top_k",
    "audio_temperature", "audio_top_p", "audio_top_k",
    "audio_repetition_penalty",
}

def test_preset_names_include_custom_first():
    assert PRESET_NAMES[0] == "Custom"
    assert "Fast (8 RVQ)" in PRESET_NAMES
    assert "Balanced (16 RVQ)" in PRESET_NAMES
    assert "High Quality (24 RVQ)" in PRESET_NAMES
    assert "Maximum (32 RVQ)" in PRESET_NAMES

def test_all_named_presets_have_same_keys():
    for name in PRESET_NAMES:
        if name == "Custom":
            continue
        assert set(PRESETS[name].keys()) == REQUIRED_KEYS, name

def test_preset_n_vq_matches_label():
    assert PRESETS["Fast (8 RVQ)"]["n_vq"] == 8
    assert PRESETS["Balanced (16 RVQ)"]["n_vq"] == 16
    assert PRESETS["High Quality (24 RVQ)"]["n_vq"] == 24
    assert PRESETS["Maximum (32 RVQ)"]["n_vq"] == 32

def test_resolve_preset_custom_returns_widgets():
    widgets = {
        "n_vq": 12,
        "text_temperature": 0.7,
        "text_top_p": 0.9,
        "text_top_k": 40,
        "audio_temperature": 0.8,
        "audio_top_p": 0.95,
        "audio_top_k": 50,
        "audio_repetition_penalty": 1.05,
    }
    assert resolve_preset("Custom", widgets) == widgets

def test_resolve_preset_named_returns_preset_values():
    widgets = {k: -1 for k in REQUIRED_KEYS}
    result = resolve_preset("Fast (8 RVQ)", widgets)
    assert result["n_vq"] == 8
    assert result["text_temperature"] == 1.5

def test_resolve_preset_unknown_raises():
    with pytest.raises(KeyError):
        resolve_preset("nonexistent", {})
