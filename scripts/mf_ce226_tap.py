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

"""Observe archived MF CE internal tensors without replacing native operations."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
from mindformers.parallel_core.training_graph.loss_func import (
    CrossEntropyLoss, _LogSoftmax, _NLLLoss, Exp, ExpandDims, F, mstype)

import capture_mf_muon
import mf_head226_tap as consumer

KINDS = ('logits', 'targets', 'mask', 'shifted', 'exponent', 'sum', 'log_sum',
         'log_probability', 'probability', 'one_hot', 'dout', 'dx_fp32', 'dx_bf16')


class ObservedCrossEntropy(CrossEntropyLoss):
    """Preserve the full parent contract including masks, monitoring and denominators."""

    def construct(self, logits, label, input_mask):
        ops.TensorDump()(self.probe_path + '_logits', ops.gather(logits, self.probe_rows, 0))
        ops.TensorDump()(self.probe_path + '_targets', label.astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_mask', ops.reshape(input_mask, (-1,)).astype(ms.float32))
        return super().construct(logits, label, input_mask)


class ObservedLogSoftmax(_LogSoftmax):
    """Keep archived max/sub/exp/sum/log/sub operations and their original full shapes."""

    def construct(self, logits, label):
        logits = self.cast(logits, mstype.float32)
        _, logit_max = self.max(logits)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1, True)
        log_exp_sum = self.log(exp_sum)
        log_softmax_result = self.sub(logit_sub, log_exp_sum)
        num_classes = F.shape(logits)[-1]
        one_hot_label = self.onehot(label, num_classes, self.on_value, self.off_value)
        ops.TensorDump()(self.probe_path + '_shifted', ops.gather(logit_sub, self.probe_rows, 0))
        ops.TensorDump()(self.probe_path + '_exponent', ops.gather(logit_exp, self.probe_rows, 0))
        ops.TensorDump()(self.probe_path + '_sum', ops.gather(exp_sum, self.probe_rows, 0))
        ops.TensorDump()(self.probe_path + '_log_sum', ops.gather(log_exp_sum, self.probe_rows, 0))
        ops.TensorDump()(self.probe_path + '_log_probability',
                         ops.gather(log_softmax_result, self.probe_rows, 0))
        return log_softmax_result, one_hot_label


class ObservedNLLLoss(_NLLLoss):
    """Keep archived custom derivative and observe its consumed dout and FP32 result."""

    def bprop(self, log_softmax_result, one_hot_label, _, dout):
        softmax_result = Exp()(log_softmax_result)
        logits = softmax_result - one_hot_label
        logits = logits * ExpandDims()(dout, -1) * self.repeat_loss
        ops.TensorDump()(self.probe_path + '_probability', ops.gather(softmax_result, self.probe_rows, 0))
        ops.TensorDump()(self.probe_path + '_one_hot',
                         ops.gather(one_hot_label, self.probe_rows, 0).astype(ms.float32))
        ops.TensorDump()(self.probe_path + '_dout',
                         ops.gather(ops.reshape(dout, (-1, 1)), self.probe_rows, 0))
        ops.TensorDump()(self.probe_path + '_dx_fp32', ops.gather(logits, self.probe_rows, 0))
        ops.TensorDump()(self.probe_path + '_dx_bf16',
                         ops.gather(logits, self.probe_rows, 0).astype(ms.bfloat16).astype(ms.float32))
        return logits, self.zeros_like(one_hot_label)


def attach(network, directory: Path) -> None:
    """Bind exactly two CE instances, leaving archive files unchanged."""
    directory.mkdir(parents=True, exist_ok=False)
    selection_path = Path('/alignment/phase12/evidence/head226-selected-coordinates.json')
    selection = json.loads(selection_path.read_text())
    rows = Tensor(np.asarray(selection['token_rows'], dtype=np.int32))
    matches = [(name, module) for name, module in network.cells_and_names()
               if isinstance(module, CrossEntropyLoss)]
    if len(matches) != 2:
        raise ValueError('Expected exactly two native cross entropy cells')
    manifest = []
    for name, module in matches:
        prefix = name.replace('.', '_')
        module.__class__ = ObservedCrossEntropy
        module._log_softmax.__class__ = ObservedLogSoftmax
        module._nllloss.__class__ = ObservedNLLLoss
        for cell in (module, module._log_softmax, module._nllloss):
            cell.probe_path = str(directory / prefix)
            cell.probe_rows = rows
        manifest.append({'name': name, 'prefix': prefix, 'repeat_loss': module._nllloss.repeat_loss})
    consumer.CONTEXT.update(directory=directory, modules=manifest)
    original_file = Path(inspect.getfile(CrossEntropyLoss))
    (directory / 'manifest.json').write_text(json.dumps({
        'modules': manifest, 'selection': selection,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'archive_source_sha256': hashlib.sha256(original_file.read_bytes()).hexdigest(),
        'original_log_softmax': inspect.getsource(_LogSoftmax.construct),
        'original_nll_backward': inspect.getsource(_NLLLoss.bprop),
        'teacher_values_used': False}, indent=2))


# Reuse the audited asynchronous-file consumer, changing only its explicit observation schema.
consumer_source = inspect.getsource(consumer.observed_end)
old = "('x', 'logits', 'dy_output', 'dy_matmul', 'dx')"
if consumer_source.count(old) != 1:
    raise ValueError('MF observation consumer changed')
consumer_source = consumer_source.replace(old, repr(KINDS)).replace('HEAD226', 'CE226')
exec(compile(consumer_source, __file__ + ':observed_end', 'exec'), consumer.__dict__)
capture_mf_muon.attach_taps = attach
capture_mf_muon.ObserveMetrics.on_train_step_end = consumer.observed_end
