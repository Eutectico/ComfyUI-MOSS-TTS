from unittest.mock import MagicMock, patch
import pytest

from lib import model_state


@pytest.fixture(autouse=True)
def clear_cache():
    model_state._MODEL_CACHE.clear()
    yield
    model_state._MODEL_CACHE.clear()


def _patches():
    """Patch transformers AutoModel/AutoProcessor with magic mocks."""
    auto_model = patch("lib.model_state.AutoModel")
    auto_proc = patch("lib.model_state.AutoProcessor")
    return auto_model, auto_proc


def test_get_or_load_creates_cache_entry():
    auto_model_p, auto_proc_p = _patches()
    with auto_model_p as MockModel, auto_proc_p as MockProc:
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
    auto_model_p, auto_proc_p = _patches()
    with auto_model_p as MockModel, auto_proc_p as MockProc:
        MockModel.from_pretrained.return_value = MagicMock()
        MockProc.from_pretrained.return_value = MagicMock()
        e1 = model_state.get_or_load("x", "cpu", "fp32", "eager")
        e2 = model_state.get_or_load("x", "cpu", "fp32", "eager")
        assert e1 is e2
        assert MockModel.from_pretrained.call_count == 1


def test_different_keys_create_distinct_entries():
    auto_model_p, auto_proc_p = _patches()
    with auto_model_p as MockModel, auto_proc_p as MockProc:
        MockModel.from_pretrained.side_effect = [MagicMock(), MagicMock()]
        MockProc.from_pretrained.side_effect = [MagicMock(), MagicMock()]
        e1 = model_state.get_or_load("x", "cpu", "fp32", "eager")
        e2 = model_state.get_or_load("x", "cpu", "fp16", "eager")
        assert e1 is not e2
        assert MockModel.from_pretrained.call_count == 2


def test_cleanup_all_clears_cache():
    auto_model_p, auto_proc_p = _patches()
    with auto_model_p as MockModel, auto_proc_p as MockProc:
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
