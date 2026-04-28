"""Module-level cache of loaded MOSS-TTS models.

Keys are (model_id, device, dtype_str, attn_impl) tuples. Cache survives across
ComfyUI workflow executions but is cleared on process exit via atexit.
"""

import atexit
import gc
import logging
from typing import Optional

import torch

log = logging.getLogger(__name__)


def _ensure_transformers_compat():
    """Patch the transformers namespace for MOSS-TTS' remote-code expectations.

    MOSS-TTS' processor and audio tokenizer (loaded via trust_remote_code=True)
    target a future transformers API. Concretely:

      1. processing_moss_tts.py does:
             processing_utils.MODALITY_TO_BASE_CLASS_MAPPING["audio_tokenizer"] = "PreTrainedModel"
         The dict only exists in newer transformers releases.

      2. configuration_moss_audio_tokenizer.py does:
             from transformers.configuration_utils import PreTrainedConfig
         Older releases call the same class `PretrainedConfig` (lowercase t).

    Rather than pinning a specific transformers version (which would conflict
    with whatever ComfyUI's other custom nodes need), we shim the missing
    names. Both shims are no-ops on transformers releases that already have
    the new API.
    """
    from transformers import processing_utils, configuration_utils
    if not hasattr(processing_utils, "MODALITY_TO_BASE_CLASS_MAPPING"):
        processing_utils.MODALITY_TO_BASE_CLASS_MAPPING = {}
    if not hasattr(configuration_utils, "PreTrainedConfig") and hasattr(configuration_utils, "PretrainedConfig"):
        configuration_utils.PreTrainedConfig = configuration_utils.PretrainedConfig
    # MOSS-TTS' processor declares `audio_tokenizer_class = "AutoModel"`, but
    # transformers' AUTO_TO_BASE_CLASS_MAPPING (which translates Auto* factory
    # names to their base class for isinstance checks) doesn't include AutoModel
    # in older releases. Without the mapping, the check resolves to the AutoModel
    # factory itself — and MossAudioTokenizerModel is not a subclass of that.
    # Adding the entry makes the check resolve to PreTrainedModel, which the
    # tokenizer model does subclass.
    if hasattr(processing_utils, "AUTO_TO_BASE_CLASS_MAPPING"):
        processing_utils.AUTO_TO_BASE_CLASS_MAPPING.setdefault("AutoModel", "PreTrainedModel")
    # Newer transformers releases auto-derive ProcessorMixin.attributes from
    # the subclass' `*_class` declarations. 4.57.6 still uses a hardcoded
    # default of ["feature_extractor", "tokenizer"]. MOSS-TTS' processor only
    # declares `tokenizer_class` and `audio_tokenizer_class` and never overrides
    # `attributes`, so __init__ wrongly insists on a feature_extractor and
    # crashes with "This processor requires 2 arguments". We monkey-patch
    # ProcessorMixin.__init__ to derive `attributes` for any subclass that
    # didn't set its own.
    # Restore _get_initial_cache_position on GenerationMixin. transformers 5.x
    # removed this method in favour of new cache helpers, but MOSS-TTS' _sample
    # still calls it as `self._get_initial_cache_position(cur_len, device, model_kwargs)`.
    from transformers.generation.utils import GenerationMixin
    if not hasattr(GenerationMixin, "_get_initial_cache_position"):
        def _get_initial_cache_position(self, cur_len, device, model_kwargs):
            if model_kwargs.get("cache_position") is None:
                past_length = 0
                past_key_values = model_kwargs.get("past_key_values")
                if past_key_values is not None:
                    try:
                        past_length = past_key_values.get_seq_length()
                    except (AttributeError, TypeError):
                        past_length = 0
                model_kwargs["cache_position"] = torch.arange(
                    past_length, cur_len, device=device
                )
            return model_kwargs
        GenerationMixin._get_initial_cache_position = _get_initial_cache_position

    pm = processing_utils.ProcessorMixin
    # Only install the attributes auto-derive shim on transformers releases that
    # have the old `attributes` / `optional_attributes` data classvars. Newer
    # releases (transformers 5.x+) have removed those entirely in favour of
    # methods like get_attributes(); the shim has nothing to do there and
    # touching the missing classvars would crash at module load.
    has_old_attrs_api = "attributes" in vars(pm) or "optional_attributes" in vars(pm)
    if has_old_attrs_api and not getattr(pm, "_moss_tts_attributes_patched", False):
        _orig_proc_init = pm.__init__

        def _patched_proc_init(self, *args, **kwargs):
            cls = type(self)
            if "attributes" not in cls.__dict__:  # subclass didn't override
                opt_attrs = set(getattr(cls, "optional_attributes", None) or ())
                derived = []
                for name in dir(cls):
                    if not name.endswith("_class") or name.startswith("_"):
                        continue
                    # Only DATA attributes count, not methods. ProcessorMixin's
                    # `*_class` declarations are strings naming the expected
                    # class (e.g. tokenizer_class = "AutoTokenizer"). Method
                    # names like check_argument_for_proper_class also end in
                    # "_class" but resolve to function objects — filter them.
                    value = getattr(cls, name, None)
                    if not isinstance(value, str):
                        continue
                    attr = name[: -len("_class")]
                    if attr in opt_attrs:
                        continue
                    derived.append(attr)
                if derived:
                    cls.attributes = derived
            _orig_proc_init(self, *args, **kwargs)

        pm.__init__ = _patched_proc_init
        pm._moss_tts_attributes_patched = True


def _get_auto_classes():
    """Lazy-import transformers' AutoModel and AutoProcessor.

    Why lazy: transformers' lazy loader transitively imports torchvision and
    other heavy deps. In some ComfyUI environments torch and torchvision
    versions are mismatched, which breaks the lazy chain at attribute access
    even though transformers itself imports fine. Deferring this import to
    runtime means our package stays importable; the failure surfaces only
    when the user actually loads a model.

    Tests can patch this single function to skip importing transformers
    entirely.
    """
    _ensure_transformers_compat()
    from transformers import AutoModel, AutoProcessor
    return AutoModel, AutoProcessor

_MODEL_CACHE: dict[tuple, dict] = {}
_ATEXIT_REGISTERED = False


class _ModelHandle(dict):
    """Dict-compatible container for the loaded MOSS-TTS model + processor.

    Inherits from dict so callers' [key] / .get() access patterns work, but
    overrides __repr__ to return a short string. Without this, ComfyUI's
    error reporter calls repr() on the workflow values, which hits
    transformers.ProcessorMixin.__repr__ -> to_dict() -> deepcopy() of every
    nested submodule including the audio_tokenizer's GPU weights — that
    triggers a *second* OOM during error formatting and masks the real error.
    """

    def __repr__(self) -> str:
        return (
            f"<MOSS_TTS_MODEL device={self.get('device')!r} "
            f"dtype={self.get('dtype')} attn={self.get('attn_impl')!r} "
            f"id={self.get('model_id')!r}>"
        )


def _ensure_atexit():
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(cleanup_all)
        _ATEXIT_REGISTERED = True


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[requested]


def _resolved_dtype_str(requested: str, device: str) -> str:
    """Return the canonical dtype string ('fp16'/'bf16'/'fp32') for cache keying."""
    if requested == "auto":
        return "fp16" if device == "cuda" else "fp32"
    return requested


def _fix_moss_model_config_token_ids(processor) -> None:
    """Make MossTTSDelayConfig's missing/None attrs fall through to language_config.

    Two distinct issues with MOSS-TTS' MossTTSDelayConfig:

    1. `__init__` sets `self.pad_token_id = pad_token_id`, then calls
       `super().__init__(**kwargs)` — and PretrainedConfig's initializer
       reassigns self.pad_token_id (and eos_token_id, bos_token_id) to None
       when those keys aren't in kwargs. The repo's saved config.json only
       stores these IDs under `language_config`, so the top-level fields end
       up None and MOSS-TTS' `_pad` crashes trying to assign None to a
       LongTensor.

    2. Many decoder-config attributes that transformers' generation/cache
       machinery accesses (num_hidden_layers, num_attention_heads, head_dim,
       max_position_embeddings, rope_*, etc.) live only on language_config —
       MOSS-TTS only propagates hidden_size and vocab_size to the top level.
       Accessing e.g. cfg.num_hidden_layers raises AttributeError and crashes
       generate's cache init.

    Fix:
    - Explicitly overwrite None token_ids from language_config (case 1).
    - Install a __getattr__ fallback on the config class that delegates any
      otherwise-missing attribute lookup to language_config (case 2). This
      catches num_hidden_layers and any future internal attr we haven't
      anticipated, without us having to enumerate them all.
    """
    cfg = getattr(processor, "model_config", None)
    if cfg is None:
        return
    lang = getattr(cfg, "language_config", None)
    if lang is None:
        return

    # Case 1: backfill None token_ids.
    for key in ("pad_token_id", "eos_token_id", "bos_token_id"):
        if getattr(cfg, key, None) is None and getattr(lang, key, None) is not None:
            setattr(cfg, key, getattr(lang, key))

    # Case 2: install __getattr__ fallback on the class once.
    cls = type(cfg)
    if not getattr(cls, "_moss_tts_lang_attr_fallback_patched", False):
        def _moss_lang_fallback(self, name):
            # Avoid infinite recursion for the language_config attr itself.
            if name == "language_config":
                raise AttributeError(name)
            lc = self.__dict__.get("language_config")
            if lc is not None:
                try:
                    return getattr(lc, name)
                except AttributeError:
                    pass
            raise AttributeError(name)

        cls.__getattr__ = _moss_lang_fallback
        cls._moss_tts_lang_attr_fallback_patched = True


def _cast_inputs_to_param_dtype_hook(module, args):
    """Forward pre-hook: cast floating tensor inputs to module's param dtype.

    AutoProcessor loads the audio_tokenizer in fp32. After we cast its weights
    to fp16/bf16 to save VRAM, raw fp32 wav inputs flow through Linear/Conv
    layers whose weights are now half-precision -> RuntimeError "expected
    scalar type Float but found Half". The hook fires before each leaf
    module's forward and aligns input dtype to the module's own parameters.
    """
    if not args:
        return args
    params = list(module.parameters(recurse=False))
    if not params:
        return args
    target_dtype = params[0].dtype
    new_args = tuple(
        a.to(target_dtype)
        if isinstance(a, torch.Tensor)
        and a.is_floating_point()
        and a.dtype != target_dtype
        else a
        for a in args
    )
    return new_args


def _patch_audio_tokenizer_dtype(audio_tokenizer) -> None:
    """Register the dtype-cast pre-hook on every leaf-ish module of the audio
    tokenizer. Covers both encode (voice ref) and decode paths without
    needing to know the model's submodule names.
    """
    for module in audio_tokenizer.modules():
        if list(module.parameters(recurse=False)):
            module.register_forward_pre_hook(_cast_inputs_to_param_dtype_hook)


def resolve_attn_impl(requested: str, device: str) -> str:
    if requested == "auto":
        return "sdpa" if device == "cuda" else "eager"
    if requested == "flash_attention_2":
        try:
            __import__("flash_attn")
        except ImportError:
            log.warning("flash_attention_2 requested but flash_attn not installed; falling back to sdpa")
            return "sdpa"
    return requested


def _set_cuda_backends():
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def get_or_load(
    model_id: str,
    device: str,
    dtype_str: str,
    attn_impl: str,
    keep_loaded: bool = True,
    audio_tokenizer_device: str = "auto",
) -> dict:
    """Load or fetch the cached MOSS-TTS model + processor.

    audio_tokenizer_device: where to place the audio_tokenizer (a separate
        small model used to encode reference audio and decode generated codes).
        "auto" mirrors the main model's device. "cpu" keeps it off the GPU,
        saving ~1-3 GB of VRAM at the cost of some latency at the start and
        end of each generation. Useful on memory-tight GPUs.
    """
    _ensure_atexit()
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype_str, resolved_device)
    resolved_attn = resolve_attn_impl(attn_impl, resolved_device)
    resolved_at_device = (
        resolved_device if audio_tokenizer_device == "auto" else audio_tokenizer_device
    )

    key = (
        model_id,
        resolved_device,
        _resolved_dtype_str(dtype_str, resolved_device),
        resolved_attn,
        resolved_at_device,
    )
    if key in _MODEL_CACHE:
        entry = _MODEL_CACHE[key]
        entry["keep_loaded"] = keep_loaded
        return entry

    log.info("Loading MOSS-TTS model %s on %s (dtype=%s, attn=%s)...",
             model_id, resolved_device, dtype_str, resolved_attn)
    _set_cuda_backends()

    AutoModel, AutoProcessor = _get_auto_classes()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _fix_moss_model_config_token_ids(processor)
    if hasattr(processor, "audio_tokenizer") and processor.audio_tokenizer is not None:
        # AutoProcessor.from_pretrained loads the audio_tokenizer in fp32
        # regardless of how the main model is loaded. On a 20 GB GPU this
        # alone pushes the working set past the headroom budget. Cast it to
        # the model's dtype on cuda; on CPU keep default precision since
        # fp16 ops are slow on most CPUs.
        if resolved_at_device == "cuda":
            processor.audio_tokenizer = processor.audio_tokenizer.to(
                resolved_at_device, dtype=resolved_dtype
            )
            _patch_audio_tokenizer_dtype(processor.audio_tokenizer)
        else:
            processor.audio_tokenizer = processor.audio_tokenizer.to(resolved_at_device)

    if resolved_device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation=resolved_attn,
        torch_dtype=resolved_dtype,
        low_cpu_mem_usage=True,
    ).to(resolved_device)
    model.eval()

    if resolved_device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    entry = _ModelHandle(
        model=model,
        processor=processor,
        device=resolved_device,
        dtype=resolved_dtype,
        attn_impl=resolved_attn,
        model_id=model_id,
        keep_loaded=keep_loaded,
    )
    _MODEL_CACHE[key] = entry
    return entry


def cleanup_all() -> str:
    """Drop every cached model and free GPU memory. Returns a status string."""
    if not _MODEL_CACHE:
        return "Model cache already empty."

    n = len(_MODEL_CACHE)
    for entry in list(_MODEL_CACHE.values()):
        entry["model"] = None
        entry["processor"] = None
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return f"Cleared {n} cached model(s). Free VRAM: {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB"
    return f"Cleared {n} cached model(s)."


def cleanup_one(key: tuple) -> bool:
    """Remove a single cached entry by key. Returns True if found and dropped."""
    if key not in _MODEL_CACHE:
        return False
    entry = _MODEL_CACHE.pop(key)
    entry["model"] = None
    entry["processor"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True
