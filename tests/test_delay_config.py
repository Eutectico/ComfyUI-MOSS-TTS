from lib.delay_config import DelayGenerationConfig

def test_default_construction():
    config = DelayGenerationConfig()
    assert config.n_vq_for_inference == 32
    assert config.do_samples is None
    assert isinstance(config.layers, list)
    assert len(config.layers) == 32

def test_custom_layers_passed_through():
    layers = [{"temperature": 0.5} for _ in range(5)]
    config = DelayGenerationConfig(layers=layers, do_samples=[True] * 5)
    assert config.layers == layers
    assert config.do_samples == [True] * 5

def test_inherits_from_generation_config():
    from transformers import GenerationConfig
    config = DelayGenerationConfig()
    assert isinstance(config, GenerationConfig)
