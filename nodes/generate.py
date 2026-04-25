"""ComfyUI node: MOSSTTSGenerate."""

import gc
import time

import torch
import torchaudio

from lib.audio import tts_output_to_audio_dict
from lib.delay_config import DelayGenerationConfig
from lib.generation import build_layers_config, format_status
from lib.model_state import cleanup_all
from lib.presets import PRESET_NAMES, resolve_preset


_DEFAULT_VOICE = {"path": None, "source": "none"}


class MOSSTTSGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MOSS_TTS_MODEL",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "preset": (PRESET_NAMES, {"default": "Balanced (16 RVQ)"}),
                "max_tokens": ("INT", {"default": 2500, "min": 50, "max": 5000, "step": 50}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "n_vq": ("INT", {"default": 16, "min": 8, "max": 32, "step": 1}),
                "text_temperature": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 2.0, "step": 0.05}),
                "text_top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "text_top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "audio_temperature": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 2.0, "step": 0.05}),
                "audio_top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
                "audio_top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "audio_repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 1.5, "step": 0.05}),
            },
            "optional": {
                "voice": ("MOSS_TTS_VOICE",),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status")
    FUNCTION = "generate"
    CATEGORY = "MOSS-TTS/generation"

    def generate(
        self,
        model,
        text,
        preset,
        max_tokens,
        speed,
        seed,
        n_vq,
        text_temperature,
        text_top_p,
        text_top_k,
        audio_temperature,
        audio_top_p,
        audio_top_k,
        audio_repetition_penalty,
        voice=None,
    ):
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        widgets = {
            "n_vq": n_vq,
            "text_temperature": text_temperature,
            "text_top_p": text_top_p,
            "text_top_k": text_top_k,
            "audio_temperature": audio_temperature,
            "audio_top_p": audio_top_p,
            "audio_top_k": audio_top_k,
            "audio_repetition_penalty": audio_repetition_penalty,
        }
        params = resolve_preset(preset, widgets)

        if seed != 0:
            torch.manual_seed(seed)

        voice = voice or _DEFAULT_VOICE
        m = model["model"]
        processor = model["processor"]
        device = model["device"]

        if voice["path"]:
            user_msg = processor.build_user_message(text=text, reference=[voice["path"]])
        else:
            user_msg = processor.build_user_message(text=text)
        conversations = [[user_msg]]

        batch = processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        gen_config = DelayGenerationConfig()
        gen_config.pad_token_id = processor.tokenizer.pad_token_id
        gen_config.eos_token_id = 151653
        gen_config.max_new_tokens = max_tokens
        gen_config.use_cache = True
        gen_config.do_sample = True
        gen_config.num_beams = 1
        gen_config.n_vq_for_inference = params["n_vq"]
        gen_config.do_samples = [True] * (params["n_vq"] + 1)
        gen_config.layers = build_layers_config(params)

        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        try:
            t0 = time.time()
            with torch.no_grad():
                outputs = m.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=gen_config,
                )
            gen_time = time.time() - t0
        except torch.cuda.OutOfMemoryError as e:
            raise torch.cuda.OutOfMemoryError(
                f"OOM with max_tokens={max_tokens}, n_vq={params['n_vq']}. "
                f"T4-class limits: 8 RVQ ~7200 tok, 16 RVQ ~4800 tok, "
                f"24 RVQ ~3000 tok, 32 RVQ ~2400 tok. Reduce max_tokens or RVQ. ({e})"
            )

        decoded = processor.decode(outputs)
        audio = decoded[0].audio_codes_list[0]

        if device == "cuda":
            del outputs, input_ids, attention_mask, batch, decoded
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        sample_rate = processor.model_config.sampling_rate
        if speed != 1.0:
            new_sr = int(sample_rate * speed)
            up = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)
            down = torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=sample_rate)
            audio = down(up(audio.unsqueeze(0)).squeeze(0).unsqueeze(0)).squeeze(0)

        audio_dict = tts_output_to_audio_dict(audio, sample_rate)
        audio_duration = audio.shape[0] / sample_rate
        vram = torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0.0

        status = format_status(
            text_length=len(text),
            max_tokens=max_tokens,
            gen_time_s=gen_time,
            audio_duration_s=audio_duration,
            vram_gb=vram,
            n_vq=params["n_vq"],
            voice_source=voice["source"],
        )

        if not model.get("keep_loaded", True):
            cleanup_all()

        return (audio_dict, status)
