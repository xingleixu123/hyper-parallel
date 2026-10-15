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

"""Use explicit contiguous router weight-gradient GEMM operands for archive compatibility."""
import hashlib
import json
from pathlib import Path
from types import MethodType
import torch
from torch.nn import functional as F
from capture_hp_native import Observer

class RouterLinear(torch.autograd.Function):
    """Keep FP32 forward/dinput and change only dweight operand storage."""
    @staticmethod
    def forward(ctx,values,weight):
        ctx.save_for_backward(values,weight)
        return F.linear(values,weight)
    @staticmethod
    def backward(ctx,gradient):
        values,weight=ctx.saved_tensors
        return gradient@weight,gradient.T.contiguous()@values

def forward(self,hidden_states):
    values=hidden_states.reshape(-1,self.hidden_dim).float()
    self.router_logits=RouterLinear.apply(values,self.weight.float())
    return self.router_logits

def install():
    original=Observer.on_train_begin
    def begin(self,state,**kwargs):
        original(self,state,**kwargs)
        names=[]
        for name,module in self.trainer.model.named_modules():
            if name in ('model.layers.1.mlp.gate','mtp.layers.0.transformer_layer.mlp.gate'):
                module.forward=MethodType(forward,module);names.append(name)
        if len(names)!=2:raise ValueError(names)
        (self.output/'ROUTER_WEIGHT_GRADIENT.json').write_text(json.dumps({
            'modules':names,'teacher_values_used':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'single-card archive compatibility','forward':'native F.linear',
            'backward':'dx=dy@weight; dw=dy.T.contiguous()@x'},indent=2))
    Observer.on_train_begin=begin
