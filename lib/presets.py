"""Quality presets for MOSS-TTS generation, copied from the source notebook."""

PRESETS: dict[str, dict] = {
    "Fast (8 RVQ)": {
        "n_vq": 8,
        "text_temperature": 1.5,
        "text_top_p": 1.0,
        "text_top_k": 50,
        "audio_temperature": 0.95,
        "audio_top_p": 0.95,
        "audio_top_k": 50,
        "audio_repetition_penalty": 1.1,
    },
    "Balanced (16 RVQ)": {
        "n_vq": 16,
        "text_temperature": 1.5,
        "text_top_p": 1.0,
        "text_top_k": 50,
        "audio_temperature": 0.95,
        "audio_top_p": 0.95,
        "audio_top_k": 50,
        "audio_repetition_penalty": 1.1,
    },
    "High Quality (24 RVQ)": {
        "n_vq": 24,
        "text_temperature": 1.5,
        "text_top_p": 1.0,
        "text_top_k": 50,
        "audio_temperature": 0.95,
        "audio_top_p": 0.95,
        "audio_top_k": 50,
        "audio_repetition_penalty": 1.1,
    },
    "Maximum (32 RVQ)": {
        "n_vq": 32,
        "text_temperature": 1.5,
        "text_top_p": 1.0,
        "text_top_k": 50,
        "audio_temperature": 0.95,
        "audio_top_p": 0.95,
        "audio_top_k": 50,
        "audio_repetition_penalty": 1.1,
    },
}

PRESET_NAMES: list[str] = ["Custom"] + list(PRESETS.keys())


def resolve_preset(preset_name: str, widget_values: dict) -> dict:
    """Return the parameter dict to use for generation.

    If preset_name == "Custom", returns widget_values unchanged.
    Otherwise returns the preset's parameters, ignoring widgets.
    """
    if preset_name == "Custom":
        return widget_values
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name!r}")
    return dict(PRESETS[preset_name])
