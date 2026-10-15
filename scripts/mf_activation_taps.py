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

"""Temporary graph observations for the first native 4K reference step."""

import json

import mindspore as ms
from mindspore import ops
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, SequenceParallelLinear
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.training_graph.transformer.moe.ffn import FFNGroupedGEMM, GroupedMatmul
from mindformers.parallel_core.training_graph.transformer.norm import FusedRMSNorm, RMSNorm
from mindformers.parallel_core.training_graph.transformer.moe.moe_layer import MoELayer
from mindformers.parallel_core.training_graph.transformer.moe.router import TopKRouter
from mindformers.parallel_core.training_graph.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher


def save_sample(path, value):
    rows = value.reshape((-1, value.shape[-1]))
    sample = ops.concat((rows[:128], rows[2240:2260]), axis=0).astype(ms.float32)
    ops.TensorDump()(path, sample)
    if ('decoder.final_layernorm' in path or 'mtp.layers.0.final_layernorm' in path
            or ('/deep/' in path and ('model.decoder.layers.0.' in path or '/mf-deepall/' in path))):
        ops.TensorDump()(path + '_full', value.astype(ms.float32))


def make_chunk_gradient_sink(path):
    paths = tuple(path + '_chunk_' + str(index).zfill(4) for index in range(64))
    def save_gradient(gradient):
        rows = gradient.reshape((-1, gradient.shape[-1]))
        for index in range(64):
            ops.TensorDump()(paths[index], rows[index * 64:(index + 1) * 64].astype(ms.float32))
        return gradient
    return save_gradient


def make_head_gradient_sink(path):
    def save_gradient(gradient):
        ops.TensorDump()(path, gradient.reshape((-1, gradient.shape[-1]))[2009:2029].astype(ms.float32))
        return gradient
    return save_gradient


def make_gradient_sink(path):
    """Observe a native graph gradient without replacing its calculation."""
    def save_gradient(gradient):
        ops.TensorDump()(path, gradient.astype(ms.float32))
        return gradient
    return save_gradient


class ObservedFusedRMSNorm(FusedRMSNorm):
    def construct(self, x):
        if hasattr(self, 'probe_input_gradient'):
            x = self.probe_input_gradient(x)
        result = super().construct(x)
        save_sample(self.probe_path + '_input', x)
        save_sample(self.probe_path, result)
        if hasattr(self, 'probe_output_gradient'):
            result = self.probe_output_gradient(result)
        return result


class ObservedRMSNorm(RMSNorm):
    def construct(self, x):
        if hasattr(self, 'probe_input_gradient'):
            x = self.probe_input_gradient(x)
        result = super().construct(x)
        save_sample(self.probe_path + '_input', x)
        save_sample(self.probe_path, result)
        if hasattr(self, 'probe_output_gradient'):
            result = self.probe_output_gradient(result)
        return result


class ObservedOutputHead(ColumnParallelLinear):
    def construct(self, input_, weight=None):
        result, bias = super().construct(input_, weight)
        if hasattr(self, 'probe_chunk_paths'):
            rows = result.reshape((-1, result.shape[-1]))
            for index in range(64):
                ops.TensorDump()(self.probe_chunk_paths[index], rows[index * 64:(index + 1) * 64].astype(ms.float32))
        else:
            ops.TensorDump()(self.probe_path + '_logits', result.reshape((-1, result.shape[-1]))[2009:2029].astype(ms.float32))
        result = self.probe_output_gradient(result)
        return result, bias


class ObservedColumn(ColumnParallelLinear):
    def construct(self, input_, weight=None):
        result, bias = super().construct(input_, weight)
        if hasattr(self, 'probe_linear_gradient'):
            result = self.probe_linear_gradient(result)
        save_sample(self.probe_path + "_input", input_)
        save_sample(self.probe_path, result)
        return result, bias


class ObservedRow(RowParallelLinear):
    def construct(self, input_):
        if hasattr(self, 'probe_linear_gradient'):
            input_ = self.probe_linear_gradient(input_)
        result, bias = super().construct(input_)
        if hasattr(self, 'probe_linear_output_gradient'):
            result = self.probe_linear_output_gradient(result)
        save_sample(self.probe_path + "_input", input_)
        save_sample(self.probe_path, result)
        return result, bias


class ObservedMoE(MoELayer):
    def construct(self, hidden_states, extra_loss=0., seq_chunk=None):
        result, bias, loss = super().construct(hidden_states, extra_loss, seq_chunk)
        if hasattr(self, 'probe_moe_output_gradient'):
            result = self.probe_moe_output_gradient(result)
        save_sample(self.probe_path, result)
        return result, bias, loss


class ObservedRouter(TopKRouter):
    def gating(self, inputs):
        if hasattr(self, "probe_router_input_gradient"):
            inputs = self.probe_router_input_gradient(inputs)
        result = super().gating(inputs)
        if hasattr(self, "probe_logits_gradient"):
            result = self.probe_logits_gradient(result)
        save_sample(self.probe_path, result)
        ops.TensorDump()(self.probe_path + '_logits_full', result)
        return result

    def routing(self, logits):
        result = super().routing(logits)
        if hasattr(self, 'probe_prob_gradient'):
            result = (self.probe_prob_gradient(result[0]), result[1], result[2])
        ops.TensorDump()(self.probe_path + '_probs_full', result[0])
        ops.TensorDump()(self.probe_path + '_indices_full', result[1])
        return result


class ObservedSequenceLinear(SequenceParallelLinear):
    def construct(self, input_, weight=None):
        if hasattr(self, 'probe_linear_gradient') and self.probe_linear_is_input:
            input_ = self.probe_linear_gradient(input_)
        result, bias = super().construct(input_, weight)
        if hasattr(self, 'probe_linear_gradient') and not self.probe_linear_is_input:
            result = self.probe_linear_gradient(result)
        save_sample(self.probe_path + '_input', input_)
        save_sample(self.probe_path, result)
        return result, bias


class ObservedExperts(FFNGroupedGEMM):
    def experts_forward(self, permuted_local_hidden_states, tokens_per_expert, w1, w2):
        if hasattr(self, 'probe_fc1_gradient'):
            inputs = permuted_local_hidden_states.reshape((-1, self.hidden_size))
            projected = GroupedMatmul(split_item=3, group_type=0)(
                [inputs], [w1], None, None, None, None, None, tokens_per_expert)[0]
            ops.TensorDump()(self.probe_path + '_fc1_full', projected.astype(ms.float32))
            projected = self.probe_fc1_gradient(projected)
            activated = self.swiglu(projected, -1).reshape((-1, w2.shape[1]))
            activated = self.probe_activation_gradient(activated)
            result = GroupedMatmul(split_item=3, group_type=0)(
                [activated], [w2], None, None, None, None, None, tokens_per_expert)[0]
        else:
            result = super().experts_forward(permuted_local_hidden_states, tokens_per_expert, w1, w2)
        if hasattr(self, 'probe_experts_gradient'):
            ops.TensorDump()(self.probe_path + '_expert_major_input', permuted_local_hidden_states.astype(ms.float32))
            ops.TensorDump()(self.probe_path + '_expert_counts', tokens_per_expert)
            result = self.probe_experts_gradient(result)
        return result

    def construct(self, tokens, probs, routing_map):
        result = super().construct(tokens, probs, routing_map)
        save_sample(self.probe_path, result)
        return result


class ObservedDispatcher(MoEAlltoAllTokenDispatcher):
    def token_unpermutation(self, tokens, ctx):
        # The diagnostic baseline has one EP rank, so its AllToAll is identity.
        outer_indices = ctx[2].reshape((1, -1, self.moe_router_topk))[:, 2248:2268].reshape(-1)
        indices = ops.gather(ctx[1].reshape(-1), outer_indices, 0)
        selected = ops.gather(tokens.reshape((-1, self.hidden_size)), indices, 0)
        ops.TensorDump()(self.probe_path + '_selected_expert_values', selected.astype(ms.float32))
        return super().token_unpermutation(tokens, ctx)


class ObservedAttention(FlashAttention):
    def construct(self, query, key, value, attention_mask, attn_mask_type=None,
                  attention_bias=None, packed_seq_params=None, alibi_mask=None,
                  prefix=None, padding_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None):
        if hasattr(self, 'probe_query_gradient'):
            query = self.probe_query_gradient(query)
            key = self.probe_key_gradient(key)
            value = self.probe_value_gradient(value)
        result = super().construct(query, key, value, attention_mask, attn_mask_type,
                                   attention_bias, packed_seq_params, alibi_mask, prefix,
                                   padding_mask, actual_seq_qlen, actual_seq_kvlen)
        if hasattr(self, 'probe_attention_output_gradient'):
            result = self.probe_attention_output_gradient(result)
        ops.TensorDump()(self.probe_path + '_query', query[:128].astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_key', key[:128].astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_value', value[:128].astype(ms.float32))
        ops.TensorDump()(self.probe_path, result[:128].astype(ms.float32))
        if '/deep/' in self.probe_path and ('model.decoder.layers.0.' in self.probe_path or '/mf-deepall/' in self.probe_path):
            ops.TensorDump()(self.probe_path + '_query_full', query.astype(ms.float32))
            ops.TensorDump()(self.probe_path + '_key_full', key.astype(ms.float32))
            ops.TensorDump()(self.probe_path + '_value_full', value.astype(ms.float32))
            ops.TensorDump()(self.probe_path + '_full', result.astype(ms.float32))
        return result


def attach_taps(network, output):
    output.mkdir()
    replacements = {FusedRMSNorm: ObservedFusedRMSNorm, RMSNorm: ObservedRMSNorm,
                    ColumnParallelLinear: ObservedColumn, RowParallelLinear: ObservedRow,
                    MoELayer: ObservedMoE, TopKRouter: ObservedRouter,
                    SequenceParallelLinear: ObservedSequenceLinear,
                    FFNGroupedGEMM: ObservedExperts, FlashAttention: ObservedAttention}
    cells = []
    for name, cell in network.cells_and_names():
        old_type = type(cell)
        cells.append({'name': name, 'class': old_type.__module__ + '.' + old_type.__name__})
        if old_type is ColumnParallelLinear and name.endswith('output_layer') and 'deepce' in str(output):
            cell.__class__ = ObservedOutputHead
            cell.probe_path = str(output / name)
            if 'deepcechunks' in str(output):
                cell.probe_chunk_paths = tuple(cell.probe_path + '_logits_chunk_' + str(index).zfill(4) for index in range(64))
                cell.probe_output_gradient = ops.InsertGradientOf(make_chunk_gradient_sink(cell.probe_path + '_gradient'))
            else:
                cell.probe_output_gradient = ops.InsertGradientOf(make_head_gradient_sink(cell.probe_path + '_gradient'))
        if old_type in replacements and not name.endswith('output_layer'):
            cell.__class__ = replacements[old_type]
            cell.probe_path = str(output / name)
            cell.probe_linear_is_input = name.endswith('linear_fc2')
            if 'deepbackward' in str(output) and old_type in {FusedRMSNorm, RMSNorm}:
                cell.probe_input_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_input_gradient'))
                cell.probe_output_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_gradient'))
            if 'deepbackwardlatent' in str(output) and name.endswith(('linear_q_down_proj', 'linear_kv_down_proj')):
                cell.probe_linear_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_gradient'))
            if 'deepbackwarddense' in str(output) and name.endswith('linear_fc1'):
                cell.probe_linear_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_gradient'))
            if 'deepbackwarddense' in str(output) and name.endswith('linear_fc2'):
                cell.probe_linear_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_input_gradient'))
            if old_type is FlashAttention and 'deepbackwardattention' in str(output) and 'mtp.layers.0.' in name:
                cell.probe_query_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_query_gradient'))
                cell.probe_key_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_key_gradient'))
                cell.probe_value_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_value_gradient'))
                cell.probe_attention_output_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_output_gradient'))
            if old_type is MoELayer and 'deepbackwardbranch' in str(output):
                cell.probe_moe_output_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_gradient'))
            if old_type is RowParallelLinear and 'deepbackwardresidual' in str(output):
                cell.probe_linear_output_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_gradient'))
            if old_type is TopKRouter and 'deepbackwardroute' in str(output):
                cell.probe_prob_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_prob_gradient'))
            if old_type is TopKRouter and 'deepbackwardrouterlogit' in str(output):
                cell.probe_logits_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_logits_gradient'))
                cell.probe_router_input_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_input_gradient'))
            if old_type is FFNGroupedGEMM:
                if 'deepbackwardexpert' in str(output) and 'mtp.layers.0.' in name:
                    cell.probe_fc1_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_fc1_gradient'))
                    cell.probe_activation_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_activation_gradient'))
                if 'deepbackwardmoe' in str(output):
                    cell.probe_experts_gradient = ops.InsertGradientOf(make_gradient_sink(cell.probe_path + '_expert_major_gradient'))
                cell.token_dispatcher.__class__ = ObservedDispatcher
                cell.token_dispatcher.probe_path = str(output / name)
    (output / 'cells.json').write_text(json.dumps(cells, indent=2))
