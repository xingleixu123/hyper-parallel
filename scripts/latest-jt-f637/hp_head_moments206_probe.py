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

"""Observe actual HP output-head moment slots after the206 update."""
import capture_hp_native
import head_moments206_observation as observation


def install() -> None:
    """Resolve the unique optimizer and parameter group that own the shared head."""
    original = capture_hp_native.Observer.on_step_end

    def end(self, state, **kwargs):
        original(self, state, **kwargs)
        if int(state.global_step) != 206:
            return
        parameter = dict(self.trainer.model.named_parameters())['lm_head.weight']
        matches = [leaf for leaf in capture_hp_native.leaves(self.trainer.optimizer) if parameter in leaf.state]
        if len(matches) != 1:
            raise ValueError('Expected unique optimizer ownership for the head')
        optimizer = matches[0]
        groups = [group for group in optimizer.param_groups if any(value is parameter for value in group['params'])]
        if len(groups) != 1:
            raise ValueError('Expected unique AdamW group for the head')
        current = optimizer.state[parameter]
        records = []
        for kind in ['exp_avg', 'exp_avg_sq']:
            records.append(observation.save(self.output, kind, capture_hp_native.numpy_value(current[kind])))
        metadata = {key: groups[0][key] for key in ['step', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad']}
        observation.finish(self.output, 'hp', capture_hp_native.numpy_value(parameter), records,
                           int(groups[0]['step']), __file__, optimizer_type=type(optimizer).__name__,
                           group=metadata, state_keys=sorted(current))

    capture_hp_native.Observer.on_step_end = end
