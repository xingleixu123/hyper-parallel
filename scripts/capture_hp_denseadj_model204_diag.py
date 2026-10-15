# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Run the independent CE addition tree, optionally composing measured gate arithmetic."""
import os
import muon_dense_adjacent_factor
import observe_hp_clipped_step205
import router_weight_contiguous_factor
import muon_three_native_npusqrt_factor
import shared_key_half_tree_factor
import force_model204_subset_diagnostic
import muon_broadcast_fp32_factor
import muon_lr_fold_factor
import muon_scalar_fp32_factor
import hashlib
import json
import muon_dense_native_tile_factor
import muon_expert_bmm_eager_factor
import mapped_reference_loader
from pathlib import Path
import observe_hp_vocab_factor
import native_clip_reduction_factor
import muon_q_native_tile_factor
import muon_ns_eager_factor
import expert_token_order_factor
import capture_hp_pr180_192
import ce_sum_tree_factor
import silu_gate_factor
import router_normalization_join_factor
import dense_up_arithmetic_factor


ORIGINAL_BEGIN = capture_hp_pr180_192.capture_hp_native.Observer.on_train_begin


def observe_begin(self, state, **kwargs):
    """Keep the original audit and record exactly which arithmetic candidate ran."""
    factor = os.environ['P12_OPTIMIZER_FACTOR']
    modules = [Path(__file__), Path(muon_dense_adjacent_factor.__file__), Path(router_weight_contiguous_factor.__file__), Path(muon_three_native_npusqrt_factor.__file__), Path(muon_dense_native_tile_factor.__file__),
               Path(muon_expert_bmm_eager_factor.__file__)]
    report = {'factor': factor, 'teacher_values_used': False,
              'control': os.environ['P12_FACTOR_CONTROL'],
              'sources': [{'name': file.name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
                          for file in modules]}
    (self.output / 'OPTIMIZER_REFINEMENT.json').write_text(json.dumps(report, indent=2))
    (self.output / "MUON_BROADCAST_PRECISION.json").write_text(json.dumps(BROADCAST_POLICY, indent=2))
    ORIGINAL_BEGIN(self, state, **kwargs)


capture_hp_pr180_192.capture_hp_native.Observer.on_train_begin = observe_begin


if __name__ == "__main__":
    router_weight_contiguous_factor.install()
    shared_key_half_tree_factor.install()
    force_model204_subset_diagnostic.install()
    muon_lr_fold_factor.install()
    muon_scalar_fp32_factor.install()
    BROADCAST_POLICY = muon_broadcast_fp32_factor.install()
    if os.environ.get("P12_MAPPED_REFERENCE") == "1":
        mapped_reference_loader.install()
    native_clip_reduction_factor.install()
    muon_ns_eager_factor.install()
    muon_three_native_npusqrt_factor.install()
    muon_q_native_tile_factor.install()
    factor = os.environ['P12_OPTIMIZER_FACTOR']
    if factor == 'dense-expert-eager-view':
        muon_dense_native_tile_factor.install()
    if factor in ('expert-eager-view', 'dense-expert-eager-view'):
        muon_expert_bmm_eager_factor.install()
    if factor not in ('expert-eager-copy', 'expert-eager-view', 'dense-expert-eager-view'):
        raise ValueError('Unknown optimizer factor')
    router_normalization_join_factor.install()
    ce_sum_tree_factor.install()
    if os.environ.get("P12_CE_TREE_SILU_GATE") == "1":
        silu_gate_factor.install()
    dense_up_arithmetic_factor.install()
    expert_token_order_factor.install()
    if os.environ.get('P12_VOCABULARY_OBSERVE') == '1':
        observe_hp_vocab_factor.install(Path('/alignment/phase12/evidence/hp-vocab-' + os.environ['P12_VOCABULARY_FACTOR'] + '.json'))
    muon_dense_adjacent_factor.install()
    capture_hp_pr180_192.capture_hp_native.main()
