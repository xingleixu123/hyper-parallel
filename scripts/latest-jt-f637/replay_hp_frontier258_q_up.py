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

"""Replay the complete actual258 update with one independent normalization factor."""
import json
from pathlib import Path

import numpy as np
import torch

import replay_hp_frontier258_muon as replay
import hp_q_up258_factor


def install_candidate_trace() -> None:
    """Record hashes only; compare complete normalization arrays without saving copies."""
    original = replay.muon_jit_probe.graph_jit_ns
    policy = hp_q_up258_factor.install()
    candidate = replay.muon_jit_probe.graph_jit_ns
    index = 0

    def traced(value, steps, normalized=None):
        nonlocal index
        piece = index
        index += 1
        normalized_candidate, scalars = hp_q_up258_factor.normalized_q_up(value)
        array = normalized_candidate.float().cpu().numpy()
        paths = list((replay.STATE / 'mf-frontier258-muon-replay').glob(
            'main1_q_up-' + str(piece) + '-normalized_*.npy'))
        if len(paths) != 1:
            raise ValueError('Expected unique normalization reference')
        expected = np.load(paths[0])
        result = candidate(value, steps, normalized)
        replay.RECORDS.append({'piece': piece, 'policy': policy, 'scalars': scalars,
                               'normalized_exact_mf': np.array_equal(array, expected),
                               'normalized_different': int(np.count_nonzero(array != expected)),
                               'normalized_sha256': replay.digest(array)})
        return result

    replay.muon_jit_probe.graph_jit_ns = traced


if __name__ == '__main__':
    replay.OUTPUT = replay.STATE / 'hp-frontier258-q-up-replay'
    replay.install_trace = install_candidate_trace
    source = Path(replay.__file__).read_text()
    source = source.replace("'evidence/frontier258-hp-muon-replay.json'",
                            "'evidence/frontier258-hp-q-up-replay.json'")
    source = source.replace("c['weight_exact_training'] and c['momentum_exact_training']",
                            "c['weight_exact_mf_training'] and c['momentum_exact_training']")
    namespace = dict(replay.__dict__)
    start = source.index('def main():')
    stop = source.index("\n\nif __name__ == '__main__':", start)
    exec(compile(source[start:stop], __file__ + ':candidate-replay', 'exec'), namespace)
    namespace['main']()
