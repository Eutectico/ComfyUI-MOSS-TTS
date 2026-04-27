# ComfyUI-MOSS-TTS

ComfyUI custom nodes for [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) — a 1.7B parameter zero-shot text-to-speech and voice-cloning model.

## Installation

1. Clone this repo into `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Eutectico/ComfyUI-MOSS-TTS.git
   ```
2. Install Python dependencies (ComfyUI does this automatically on next start, or you can run it manually):
   ```bash
   pip install -r ComfyUI-MOSS-TTS/requirements.txt
   ```
3. Restart ComfyUI. The nodes appear under the `MOSS-TTS/` category in the node menu.

The first generation downloads ~13 GB of model weights from HuggingFace. Subsequent runs use the cached weights.

### Transformers version

MOSS-TTS' remote code (loaded via `trust_remote_code=True`) is calibrated for `transformers >= 4.50, < 4.58`. Earlier releases miss newer APIs the model expects (`MODALITY_TO_BASE_CLASS_MAPPING`, `PreTrainedConfig`); later releases (especially the 5.x line) have removed APIs the model still calls (`_get_initial_cache_position`, etc.). The package contains compatibility shims that paper over a few specific gaps, but staying inside the supported range produces the cleanest results.

If the shipped audio sounds like noise rather than speech, the most common cause is that an unsupported transformers version is installed. Confirm with:
```bash
pip show transformers
```
and pin to a 4.5x version if needed.

### Language support

MOSS-TTS was trained primarily on Chinese. Without a voice reference, the default voice tends to render any input with Chinese-leaning phonetics — including English text. For English (or any non-Chinese language) output, supply a clean 5–15 second reference clip in the target language via the `MOSS-TTS Voice Reference` node.

## Nodes

| Node | Category | Purpose |
|---|---|---|
| `MOSS-TTS Loader` | `MOSS-TTS/loaders` | Loads the model + processor, holds them in a module-level cache. |
| `MOSS-TTS Voice Reference` | `MOSS-TTS/voice` | Wraps an `AUDIO` input or filesystem path into a voice-reference container. |
| `MOSS-TTS Generate` | `MOSS-TTS/generation` | Generates speech from text (with optional voice reference). |
| `MOSS-TTS Unload` | `MOSS-TTS/utility` | Frees the model from VRAM. Optional; place at the end of a workflow if needed. |

## Quality Presets

The `Generate` node has a `preset` dropdown:

| Preset | RVQ Layers | Notes |
|---|---|---|
| `Custom` | from widgets | Uses the individual sampling sliders as-is. |
| `Fast (8 RVQ)` | 8 | Longest audio on T4 (~12 min); lower quality. |
| `Balanced (16 RVQ)` | 16 | Default. ~8 min on T4. |
| `High Quality (24 RVQ)` | 24 | ~5 min on T4. |
| `Maximum (32 RVQ)` | 32 | Best quality, ~4 min on T4. |

When `preset` is anything other than `Custom`, the preset's values **override** the individual sampling widgets at execution time.

## Hardware

Defaults target T4-class GPUs (~16 GB VRAM, FP16, sdpa attention). If you have ≥24 GB VRAM, you may want to set `dtype=bf16` and increase `max_tokens`.

## Smoke test

1. Drop in a `MOSS-TTS Loader` node (defaults are fine).
2. Drop in a `MOSS-TTS Generate` node, connect `model` and type any text.
3. Connect `audio` to `Save Audio` (built-in).
4. Run. First run downloads weights (~5–8 min); subsequent runs are fast.

## Voice cloning

Connect a `Load Audio` node → `MOSS-TTS Voice Reference` → `MOSS-TTS Generate.voice`. A 5–15 second clean reference clip works well.

## Credits

- Model: [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) by OpenMOSS Team
- Source notebook: AIQUEST (`MOSS_TTS_Voice_Cloning.ipynb`)
- ComfyUI integration: this repo

## License

MIT — see `LICENSE`.
