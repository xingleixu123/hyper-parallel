"""Real HF loading against old/new frozen source; CPU only."""
import hashlib, json, sys, tempfile
from pathlib import Path
import torch
from torch import nn
from transformers import PreTrainedModel
from safetensors.torch import load_file, save_file
from hyper_parallel.models.xing_4.adapter.moe import Xing4Router
from hyper_parallel.models.xing_4.adapter.training import initialize
from hyper_parallel.models.xing_4.transformer.configuration_xing4_0 import Xing4_0Config
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import Xing4_0TopkRouter, Xing4_0HyperConnection

class ActualHF(PreTrainedModel):
    config_class = Xing4_0Config
    _keys_to_ignore_on_load_missing = [r"gate\._token_counts"]
    def __init__(self, config):
        super().__init__(config)
        self.gate = Xing4Router(config, Xing4_0TopkRouter(config))
        self.hc = Xing4_0HyperConnection(config)
        self.post_init()
    def _init_weights(self, module):
        initialize(self, module)

cfg = Xing4_0Config(hidden_size=4, n_routed_experts=4, num_experts_per_tok=2,
                    use_pad_tokens=False, moe_aux_loss_coeff=0., moe_input_jitter_eps=0.,
                    moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.001,
                    mhc_init_gating_factor=0.125)
torch.manual_seed(909)
m = ActualHF(cfg)
random_hash = {k: hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest()
               for k,v in m.state_dict().items()}
with torch.no_grad():
    m.gate.weight.fill_(7.)
    m.gate.e_score_correction_bias.fill_(3.)
    m.gate._token_counts.fill_(19)
    m.hc.hc_fn.fill_(5.)
    m.hc.hc_scale.fill_(2.)
    m.hc.hc_base.fill_(4.)
results={}
for case, missing in {
    "count": ("gate._token_counts",),
    "count_bias": ("gate._token_counts", "gate.e_score_correction_bias"),
    "all_present": (),
    "hc_fn": ("hc.hc_fn",),
}.items():
    with tempfile.TemporaryDirectory(prefix="hf-guard-") as path:
        m.save_pretrained(path, safe_serialization=True)
        file=Path(path)/"model.safetensors"
        state=load_file(str(file))
        for key in missing: del state[key]
        save_file(state, str(file), metadata={"format":"pt"})
        loaded=ActualHF.from_pretrained(path)
        results[case] = {
            "count": int(loaded.gate._token_counts[0]),
            "bias": float(loaded.gate.e_score_correction_bias[0]),
            "weight": float(loaded.gate.weight[0,0]),
            "scale": float(loaded.hc.hc_scale[0]),
            "base": float(loaded.hc.hc_base[0]),
        }
print("RESULT_JSON", json.dumps({"cases":results,"random_hash":random_hash}, sort_keys=True))
if sys.argv[1] == "new":
    assert results["count"]["count"] == 0 and results["count"]["bias"] == 3 and results["count"]["weight"] == 7
    assert results["count_bias"]["count"] == 0 and results["count_bias"]["bias"] == 0
    assert results["all_present"]["count"] == 19 and results["all_present"]["bias"] == 3
    assert results["hc_fn"]["scale"] == 2 and results["hc_fn"]["base"] == 4
