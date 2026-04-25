"""Pure helpers for assembling generation parameters and status output."""


def apply_temperature_workaround(temperature: float) -> float:
    """Workaround for the temperature=1.0 numerical bug present in MOSS-TTS.

    The notebook bumps 1.0 to 1.001 to avoid an internal divide-by-zero / NaN
    path. This is a verbatim copy of that workaround.
    """
    return 1.001 if temperature == 1.0 else temperature


def build_layers_config(params: dict) -> list[dict]:
    """Build the per-layer sampling config: 1 text layer + n_vq audio layers."""
    text_layer = {
        "repetition_penalty": 1.0,
        "temperature": apply_temperature_workaround(params["text_temperature"]),
        "top_p": params["text_top_p"],
        "top_k": params["text_top_k"],
    }
    audio_layer = {
        "repetition_penalty": params["audio_repetition_penalty"],
        "temperature": apply_temperature_workaround(params["audio_temperature"]),
        "top_p": params["audio_top_p"],
        "top_k": params["audio_top_k"],
    }
    return [text_layer] + [dict(audio_layer) for _ in range(params["n_vq"])]


def format_status(
    text_length: int,
    max_tokens: int,
    gen_time_s: float,
    audio_duration_s: float,
    vram_gb: float,
    n_vq: int,
    voice_source: str,
) -> str:
    rtf = gen_time_s / audio_duration_s if audio_duration_s > 0 else 0.0
    return (
        f"Text: {text_length} chars | "
        f"Max tokens: {max_tokens} | "
        f"Voice: {voice_source} | "
        f"RVQ: {n_vq}/32\n"
        f"Audio: {audio_duration_s:.2f}s | "
        f"Gen time: {gen_time_s:.2f}s | "
        f"RTF: {rtf:.2f}x | "
        f"VRAM: {vram_gb:.2f} GB"
    )
