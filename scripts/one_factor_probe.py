# Copyright 2026 Huawei Technologies Co., Ltd
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

"""One-factor diagnostic interventions; each run preserves the native optimizer."""
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np
import torch
import torch_npu
from torch import nn

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager, _WrapModuleInfo
from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
from hyper_parallel.components.losses import calculate_mtp_loss
from hyper_parallel.components.losses.mtp import iter_mtp_targets

class PrecisionFSDP(FSDP2Manager):
    """Retain selected small precision-sensitive units in FP32."""

    def _find_wrap_modules(self, model, metadata_by_parameter=None):
        units = super()._find_wrap_modules(model, metadata_by_parameter)
        selected = {id(unit.module) for unit in units}
        for name, module in dict(model.named_modules()).items():
            if getattr(module, "keep_compute_in_fp32", False) and id(module) not in selected:
                units.append(_WrapModuleInfo(name, module))
                selected.add(id(module))
        return units

    def _parallelize_child_units(self, wrap_modules, owner_by_parameter, metadata_by_parameter,
                                replicate_params, dense_fsdp_kwargs):
        precision = dict(dense_fsdp_kwargs)
        precision["mp_policy"] = replace(dense_fsdp_kwargs["mp_policy"], param_dtype=torch.float32,
                                          cast_forward_inputs=False, output_dtype=None)
        embedding = dict(precision)
        embedding["mp_policy"] = replace(precision["mp_policy"],
                                          output_dtype=dense_fsdp_kwargs["mp_policy"].param_dtype)
        count = 0
        for unit in sorted(wrap_modules, key=lambda item: item.fqn.count("."), reverse=True):
            options = precision if getattr(unit.module, "keep_compute_in_fp32", False) else dense_fsdp_kwargs
            if options is precision and isinstance(unit.module, nn.Embedding):
                options = embedding
            count += super()._parallelize_child_units(
                [unit], owner_by_parameter, metadata_by_parameter, replicate_params, options)
        return count


class MFNorm(jt.DeepseekV32RMSNorm):
    """Compute native fused RMSNorm with FP32 input and affine weight, then cast."""
    keep_compute_in_fp32 = True

    def forward(self, hidden_states):
        value = torch_npu.npu_rms_norm(hidden_states.float(), self.weight.float(),
                                     epsilon=self.variance_epsilon)[0]
        return value.to(getattr(self, "alignment_output_dtype", hidden_states.dtype))


class FP32Router(jt.JTDeepseekV3Gate):
    """Retain native router arithmetic and FP32 weights through FSDP."""
    keep_compute_in_fp32 = True


class GlobalMTPModel(jt.JTDeepseekV3ForCausalLM):
    """Diagnostic compatibility with archived MF's global MTP shift."""
    def compute_jt_losses(self, input_ids: torch.Tensor, labels: torch.Tensor,
                          loss_mask: torch.Tensor, *, position_ids: torch.Tensor | None = None,
                          actual_seq_len: tuple[int, ...] | None = None, sequence_start: int = 0
                          ) -> dict[str, torch.Tensor]:
        """Compute the LM, MTP and router auxiliary losses on pre-shifted labels.

        Args:
            input_ids: Token IDs, ``[1, sequence]``.
            labels: Targets already shifted by one token.
            loss_mask: Mask of the supervised targets.
            position_ids: Optional sequence positions for RoPE.
            actual_seq_len: Global cumulative ends of independent documents.
            sequence_start: Global offset of the local input interval.

        Returns:
            ``lm_loss``, ``mtp_loss`` and ``aux_loss`` as 0-d tensors.

        Raises:
            ValueError: If the inputs are not one sequence with matching shapes.
        """
        cfg = self.config
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("JT sequence loss currently requires a two-dimensional batch-one input")
        sequence_length = input_ids.shape[1]
        if labels.shape != input_ids.shape or loss_mask.shape != input_ids.shape:
            raise ValueError("Tokens, pre-shifted labels and loss mask must have identical shapes")
        ends = (sequence_length,) if actual_seq_len is None else tuple(actual_seq_len)
        if (not ends or ends[0] <= 0 or any(left >= right for left, right in zip(ends, ends[1:]))
                or sequence_start < 0 or sequence_start + sequence_length > ends[-1]):
            raise ValueError("JT cumulative document ends must increase and cover the local token interval")
        indices = torch.arange(sequence_start, sequence_start + sequence_length, device=input_ids.device)
        sequence_end_mask = None
        if len(ends) > 1:
            boundaries = indices.new_tensor((0, *ends))
            documents = torch.bucketize(indices, boundaries[1:], right=True)
            sequence_end_mask = (indices + 1 == boundaries[documents + 1]).unsqueeze(0)
            default_positions = indices - boundaries[documents]
        else:
            default_positions = indices
        # NumPy FP32 inverse frequencies match MindFormers bitwise; torch.pow rounds some entries differently.
        sequence_end_mask = None
        dim = cfg.qk_rope_head_dim
        inverse = 1.0 / (cfg.rope_parameters["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        inverse = torch.from_numpy(inverse.astype(np.float32)).to(input_ids.device)
        if position_ids is None:
            position_ids = default_positions.unsqueeze(0)
        if position_ids.shape != input_ids.shape:
            raise ValueError("JT position_ids must match input_ids")
        frequency = position_ids.to(device=input_ids.device, dtype=torch.float32).unsqueeze(-1) * inverse
        frequency = torch.cat((frequency, frequency), dim=-1)
        hidden = self.model.embed_tokens(input_ids)
        # FP32 tables; JTDeepseekV3RotaryEmbedding rotates in FP32 and rounds to the activation dtype once.
        attention_kwargs = {"position_embeddings": (frequency.cos(), frequency.sin()), "actual_seq_len": ends}
        auxiliary = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for layer in self.model.layers:
            hidden = layer(hidden, **attention_kwargs)
            if hasattr(layer.mlp, "auxiliary_loss"):
                auxiliary = auxiliary + layer.mlp.auxiliary_loss
        targets = labels.masked_fill(loss_mask == 0, -100)
        normalized = self.model.norm(hidden)
        lm_loss = (self._projection_loss(normalized, targets) if self.loss_chunk_size
                   else self._token_loss(self.lm_head(normalized), labels, loss_mask))
        mtp_output = self.mtp(hidden, input_ids, embedding=self.model.embed_tokens,
                              head=None if self.loss_chunk_size else self.lm_head,
                              decoder_kwargs=attention_kwargs, shift_fn=self.shift_mtp_inputs,
                              sequence_end_mask=sequence_end_mask)
        for layer in self.mtp.layers:
            auxiliary = auxiliary + layer.transformer_layer.mlp.auxiliary_loss
        shift_targets = partial(self.shift_mtp_inputs, pad_value=-100)
        if self.loss_chunk_size:
            mtp_loss = torch.zeros_like(lm_loss)
            depths = len(mtp_output.prediction_hidden_states)
            targets_by_depth = iter_mtp_targets(targets, depths, shift_fn=shift_targets,
                                                sequence_end_mask=sequence_end_mask)
            for prediction, future_targets in zip(mtp_output.prediction_hidden_states, targets_by_depth):
                mtp_loss = mtp_loss + self._projection_loss(prediction, future_targets) * (cfg.mtp_loss_factor / depths)
        else:
            mtp_loss = calculate_mtp_loss(mtp_output.logits, targets, self._mtp_token_loss, vocab_size=cfg.vocab_size,
                                          loss_factor=cfg.mtp_loss_factor, shift_fn=shift_targets,
                                          sequence_end_mask=sequence_end_mask)
        return {"lm_loss": lm_loss, "mtp_loss": mtp_loss, "aux_loss": auxiliary}


def configure_factor(model: torch.nn.Module, factor: str) -> None:
    """Change exactly one requested computational factor."""
    changed = 0
    for module in model.modules():
        if factor == "norm-fp32" and type(module) is jt.DeepseekV32RMSNorm:
            module.__class__ = MFNorm
            changed += 1
        elif factor == "router-fp32" and type(module) is jt.JTDeepseekV3Gate:
            module.__class__ = FP32Router
            changed += 1
    if factor in ("norm-fp32", "router-fp32") and not changed:
        raise ValueError("No intended modules changed")
    print("[P12-FACTOR]", factor, "changed_modules", changed, flush=True)
