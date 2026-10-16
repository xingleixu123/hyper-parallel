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

"""Compare MTP reduction precision in memory; retain only hashes and aggregate errors."""
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from capture_hp_native import Observer
import expert_token_order_factor
import hp_moe236_hashes as base
from mindspore272_stat_hash import native_dump_sha1

CONTEXT: dict[str, Any] = {'arrays': {}, 'attached': False}


def bf16(value: np.ndarray) -> np.ndarray:
    """Round finite FP32 values to BF16, retaining an exactly representable FP32 view."""
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def bf16_hash(value: np.ndarray) -> str:
    """Encode the native two-byte BF16 representation for MindSpore Dump comparison."""
    words = (np.ascontiguousarray(bf16(value)).view(np.uint32) >> 16).astype(np.uint16)
    return native_dump_sha1(words)


def tree_sum(product: np.ndarray) -> np.ndarray:
    """Replay the separately verified 80-row odd-tail tree and 64 adjacent lanes in FP32."""
    work = product.reshape(-1, 80, 64).transpose(0, 2, 1)
    while work.shape[-1] > 1:
        size = work.shape[-1]
        half = size // 2
        reduced = work[..., :half] + work[..., half:2 * half]
        if size % 2:
            reduced[..., 0] = reduced[..., 0] + work[..., -1]
        work = reduced
    lanes = work[..., 0]
    while lanes.shape[-1] > 1:
        lanes = lanes[..., ::2] + lanes[..., 1::2]
    return lanes.reshape(product.shape[:-1])


def precision_report() -> dict[str, Any]:
    """Canonicalize expert order, verify actual derivatives, and compare high precision sums."""
    arrays = CONTEXT['arrays']
    order = arrays['order'].reshape(-1)
    if not np.array_equal(np.sort(order), np.arange(order.size)):
        raise ValueError('Incomplete expert permutation')
    inverse = np.argsort(order)
    values = arrays['weighted.outputs'][inverse].reshape(-1, 2, 5120)
    gradient = arrays['weighted.dy'][inverse].reshape(-1, 2, 5120)
    weights = arrays['weighted.weights'][inverse].reshape(-1, 2)
    actual = arrays['weighted.weights_dy'][inverse].reshape(-1, 2)
    coefficients = arrays['normalization.coefficients_dy'].reshape(-1, 2)
    padding = values.shape[0] - coefficients.shape[0]
    if padding != 8 or values.shape[0] != 4104:
        raise ValueError('Unexpected native padded token layout')
    if not np.array_equal(gradient[:, 0].view(np.uint32), gradient[:, 1].view(np.uint32)):
        raise ValueError('Expert-choice upstream gradients differ')
    if not np.array_equal(actual[padding:].view(np.uint32), coefficients.view(np.uint32)):
        raise ValueError('Inverse expert permutation does not reproduce actual coefficient gradients')
    product = bf16(bf16(gradient) * values)
    reference = np.sum(product, axis=-1, dtype=np.float64)
    candidate = bf16(tree_sum(product))
    changed = actual.view(np.uint32) != candidate.view(np.uint32)
    hp_error = np.abs(actual.astype(np.float64) - reference)
    mf_candidate_error = np.abs(candidate.astype(np.float64) - reference)
    scopes = {}
    for name, selection in [('including_padding', slice(None)), ('real_tokens', slice(padding, None))]:
        different = changed[selection]
        left, right = hp_error[selection], mf_candidate_error[selection]
        scopes[name] = {
            'choices': int(different.size), 'different': int(np.count_nonzero(different)),
            'max_candidate_actual_abs_difference': float(np.max(np.abs(actual[selection] - candidate[selection]))),
            'hp_closer_at_differences': int(np.count_nonzero(different & (left < right))),
            'candidate_closer_at_differences': int(np.count_nonzero(different & (right < left))),
            'ties_at_differences': int(np.count_nonzero(different & (left == right))),
            'hp_max_abs_error': float(left.max()), 'candidate_max_abs_error': float(right.max()),
        }
    return {
        'complete': True, 'step': 236, 'padding': padding, 'raw_arrays_written': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_observer_sha256': hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        'hash_helper_sha256': hashlib.sha256(Path(__file__).with_name('mindspore272_stat_hash.py').read_bytes()).hexdigest(),
        'canonical_hashes': {
            'weighted.outputs_bf16': bf16_hash(values),
            'weighted.dy_bf16_single_choice': bf16_hash(gradient[:, 0]),
            'weighted.weights_bf16': bf16_hash(weights),
            'actual_coefficients_dy_fp32': native_dump_sha1(coefficients),
            'candidate_coefficients_dy_fp32': native_dump_sha1(np.ascontiguousarray(candidate[padding:])),
            'fp64_reference_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
        },
        'precision': scopes,
        'scope': 'FP64 sum of actual BF16-rounded products. Candidate equals archived MF only after separate native '
                 'hash and trajectory fidelity audit; high precision comparison is restricted to this input.',
    }


def install() -> None:
    """Extend the read-only boundary observer and never return replay values to training."""
    base.install()
    original_save = base.save
    original_begin, original_end = Observer.on_micro_step_begin, Observer.on_step_end

    def save(name: str, value: Any) -> None:
        original_save(name, value)
        if name not in {'weighted.outputs', 'weighted.weights', 'weighted.dy', 'weighted.weights_dy',
                        'normalization.coefficients_dy'}:
            return
        array = value.detach().float().cpu().numpy()
        CONTEXT['arrays'][name] = array

    def begin(self: Any, state: Any, micro_batch: Any, **kwargs: Any) -> None:
        original_begin(self, state, micro_batch, **kwargs)
        if int(state.global_step) != 235 or CONTEXT['attached']:
            return
        CONTEXT['attached'] = True
        original_order = expert_token_order_factor.expert_order

        def order(indices: Any) -> Any:
            result = original_order(indices)
            if base.CONTEXT['active']:
                array = result.detach().cpu().numpy()
                prior = CONTEXT['arrays'].get('order')
                if prior is not None and not np.array_equal(prior, array):
                    raise ValueError('Recomputation changed expert order')
                CONTEXT['arrays']['order'] = array
            return result

        expert_token_order_factor.expert_order = order

    def end(self: Any, state: Any, **kwargs: Any) -> None:
        original_end(self, state, **kwargs)
        if int(state.global_step) == 236:
            report = precision_report()
            (self.output / 'WEIGHTED236_PRECISION.json').write_text(json.dumps(report, indent=2))
            CONTEXT['arrays'].clear()

    base.save = save
    Observer.on_micro_step_begin, Observer.on_step_end = begin, end
