"""Custom GenerationConfig for MOSS-TTS multi-layer RVQ generation.

Verbatim from the source notebook (MOSS_TTS_Voice_Cloning.ipynb cell 8).
Required by MOSS-TTS' generate() to specify per-layer sampling parameters
across all 32 RVQ codebook layers plus the text layer.
"""

from transformers import GenerationConfig


class DelayGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = kwargs.get("layers", [{} for _ in range(32)])
        self.do_samples = kwargs.get("do_samples", None)
        self.n_vq_for_inference = 32
