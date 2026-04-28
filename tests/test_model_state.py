from unittest.mock import MagicMock, patch
import pytest

from lib import model_state


@pytest.fixture(autouse=True)
def clear_cache():
    model_state._MODEL_CACHE.clear()
    yield
    model_state._MODEL_CACHE.clear()


def _patch_classes():
    """Patch the lazy-import accessor with mocked AutoModel/AutoProcessor.

    Returns the patch context, plus the mock classes so tests can configure
    their `.from_pretrained` return values.
    """
    fake_model = MagicMock()
    fake_proc = MagicMock()
    return (
        patch("lib.model_state._get_auto_classes", return_value=(fake_model, fake_proc)),
        fake_model,
        fake_proc,
    )


def test_get_or_load_creates_cache_entry():
    p, MockModel, MockProc = _patch_classes()
    with p:
        MockModel.from_pretrained.return_value = MagicMock()
        MockProc.from_pretrained.return_value = MagicMock()
        entry = model_state.get_or_load(
            model_id="x", device="cpu", dtype_str="fp32", attn_impl="eager"
        )
        assert "model" in entry
        assert "processor" in entry
        assert entry["device"] == "cpu"
        MockModel.from_pretrained.assert_called_once()
        MockProc.from_pretrained.assert_called_once()


def test_get_or_load_returns_cached_entry_on_second_call():
    p, MockModel, MockProc = _patch_classes()
    with p:
        MockModel.from_pretrained.return_value = MagicMock()
        MockProc.from_pretrained.return_value = MagicMock()
        e1 = model_state.get_or_load("x", "cpu", "fp32", "eager")
        e2 = model_state.get_or_load("x", "cpu", "fp32", "eager")
        assert e1 is e2
        assert MockModel.from_pretrained.call_count == 1


def test_different_keys_create_distinct_entries():
    p, MockModel, MockProc = _patch_classes()
    with p:
        MockModel.from_pretrained.side_effect = [MagicMock(), MagicMock()]
        MockProc.from_pretrained.side_effect = [MagicMock(), MagicMock()]
        e1 = model_state.get_or_load("x", "cpu", "fp32", "eager")
        e2 = model_state.get_or_load("x", "cpu", "fp16", "eager")
        assert e1 is not e2
        assert MockModel.from_pretrained.call_count == 2


def test_cleanup_all_clears_cache():
    p, MockModel, MockProc = _patch_classes()
    with p:
        MockModel.from_pretrained.return_value = MagicMock()
        MockProc.from_pretrained.return_value = MagicMock()
        model_state.get_or_load("x", "cpu", "fp32", "eager")
        assert len(model_state._MODEL_CACHE) == 1
        model_state.cleanup_all()
        assert len(model_state._MODEL_CACHE) == 0


def test_resolve_dtype_auto_picks_fp16_on_cuda():
    import torch
    assert model_state.resolve_dtype("auto", "cuda") == torch.float16
    assert model_state.resolve_dtype("auto", "cpu") == torch.float32
    assert model_state.resolve_dtype("bf16", "cpu") == torch.bfloat16
    assert model_state.resolve_dtype("fp32", "cuda") == torch.float32


def test_resolve_attn_impl_auto():
    assert model_state.resolve_attn_impl("auto", "cuda") == "sdpa"
    assert model_state.resolve_attn_impl("auto", "cpu") == "eager"
    assert model_state.resolve_attn_impl("sdpa", "cuda") == "sdpa"


def test_resolve_attn_impl_flash_falls_back_when_unavailable(monkeypatch):
    # Simulate flash_attn not being importable
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert model_state.resolve_attn_impl("flash_attention_2", "cuda") == "sdpa"


def test_auto_dtype_collapses_with_explicit_on_same_device():
    """auto and the matching explicit dtype should hit the same cache entry."""
    p, MockModel, MockProc = _patch_classes()
    with p, patch("lib.model_state.torch.cuda.is_available", return_value=False):
        MockModel.from_pretrained.return_value = MagicMock()
        MockProc.from_pretrained.return_value = MagicMock()
        e1 = model_state.get_or_load("x", "cpu", "auto", "eager")
        e2 = model_state.get_or_load("x", "cpu", "fp32", "eager")
        assert e1 is e2
        assert MockModel.from_pretrained.call_count == 1


class _MiniAudioTokenizer(__import__("torch").nn.Module):
    """Minimal stand-in mixing Linear, Conv1d, and LayerNorm — enough to
    exercise the mixed-precision logic without loading MOSS-TTS' full model.
    """

    def __init__(self):
        super().__init__()
        import torch
        self.linear = torch.nn.Linear(4, 4)
        self.conv = torch.nn.Conv1d(4, 4, kernel_size=1)
        self.norm = torch.nn.LayerNorm(4)


def test_mixed_precision_casts_linear_keeps_norm_fp32():
    """audio_tokenizer cast to fp16, but LayerNorm stays fp32 to preserve
    the precision normalization statistics need.
    """
    import torch
    at = _MiniAudioTokenizer()
    model_state._patch_audio_tokenizer_mixed_precision(at, torch.float16)
    assert at.linear.weight.dtype == torch.float16
    assert at.conv.weight.dtype == torch.float16
    assert at.norm.weight.dtype == torch.float32


def test_mixed_precision_hook_lets_fp32_input_flow_through_fp16_linear():
    """End-to-end: an fp32 wav input survives a Linear with fp16 weights via
    the dtype-cast pre-hook installed during patching.
    """
    import torch
    at = _MiniAudioTokenizer()
    model_state._patch_audio_tokenizer_mixed_precision(at, torch.float16)
    inp_fp32 = torch.randn(2, 4, dtype=torch.float32)
    out = at.linear(inp_fp32)
    assert out.dtype == torch.float16


def test_mixed_precision_hook_supplies_norm_with_fp32():
    """fp16 hidden states reach the (fp32) norm: hook casts back to fp32."""
    import torch
    at = _MiniAudioTokenizer()
    model_state._patch_audio_tokenizer_mixed_precision(at, torch.float16)
    fp16_hidden = torch.randn(2, 4, dtype=torch.float16)
    out = at.norm(fp16_hidden)
    assert out.dtype == torch.float32


def test_mixed_precision_skipped_when_audio_tokenizer_on_cpu():
    """CPU path keeps the audio_tokenizer at default precision — fp16 ops
    are slow on most CPUs and the VRAM concern doesn't apply off-GPU.
    """
    p, MockModel, MockProc = _patch_classes()
    with p, patch("lib.model_state._patch_audio_tokenizer_mixed_precision") as mock_patch:
        MockProc.from_pretrained.return_value = MagicMock()
        MockModel.from_pretrained.return_value = MagicMock()
        model_state.get_or_load(
            "x", "cuda", "fp16", "sdpa", audio_tokenizer_device="cpu"
        )
        mock_patch.assert_not_called()
