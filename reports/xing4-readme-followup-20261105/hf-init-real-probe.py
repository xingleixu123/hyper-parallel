"""Real HF PreTrainedModel + Xing4Router: missing count and sibling bias finalization."""
import torch
from torch import nn
from transformers import PreTrainedModel
from hyper_parallel.components.checkpoint.load_groups import LoadReport
from hyper_parallel.models._transformers.checkpoint_loader import _finalize_model_loading
from hyper_parallel.models.xing_4.adapter.moe import Xing4Router
from hyper_parallel.models.xing_4.adapter.training import initialize
from hyper_parallel.models.xing_4.transformer.configuration_xing4_0 import Xing4_0Config
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import Xing4_0TopkRouter

class ActualHFModel(PreTrainedModel):
    config_class = Xing4_0Config
    _keys_to_ignore_on_load_missing = [r"gate\._token_counts"]
    def __init__(self, config):
        super().__init__(config)
        self.gate = Xing4Router(config, Xing4_0TopkRouter(config))
        self.post_init()
    def _init_weights(self, module):
        initialize(self, module)

cfg = Xing4_0Config(hidden_size=4, n_routed_experts=4, num_experts_per_tok=2,
                    use_pad_tokens=False, moe_aux_loss_coeff=0., moe_input_jitter_eps=0.,
                    moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.001)
def fresh():
    model = ActualHFModel(cfg)
    with torch.no_grad():
        model.gate.weight.fill_(7.)
        model.gate.e_score_correction_bias.fill_(3.)
        model.gate._token_counts.fill_(123)
    return model

report = LoadReport(("gate.weight", "gate.e_score_correction_bias"), ("gate._token_counts",), ())
m = fresh()
_finalize_model_loading(m, report, strict=True)
print("HP ignored missing: count", m.gate._token_counts.tolist(), "bias", m.gate.e_score_correction_bias.tolist())

m = fresh()
m._keys_to_ignore_on_load_missing = set()
_finalize_model_loading(m, report, strict=False)
print("HP initializing missing: count", m.gate._token_counts.tolist(), "loaded bias", m.gate.e_score_correction_bias.tolist(),
      "loaded weight unchanged", bool((m.gate.weight == 7.).all()))

# Exercise Transformers' own from_pretrained ordering on the same actual adapter.
import tempfile
from pathlib import Path
from safetensors.torch import load_file, save_file
with tempfile.TemporaryDirectory(prefix="xing4-hf-native-") as path:
    m = fresh()
    m.save_pretrained(path, safe_serialization=True)
    file = Path(path) / "model.safetensors"
    state = load_file(str(file))
    del state["gate._token_counts"]
    save_file(state, str(file), metadata={"format": "pt"})
    loaded, info = ActualHFModel.from_pretrained(path, output_loading_info=True)
    print("HF from_pretrained: missing", info["missing_keys"], "count", loaded.gate._token_counts.tolist(),
          "bias", loaded.gate.e_score_correction_bias.tolist())
