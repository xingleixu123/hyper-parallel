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

"""Compare actual optimizer201 and model202 without writing extra checkpoints."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from capture_hp_native import Observer, leaves, numpy_value
from observe_hp_poststep_model import compare

ROOT = Path('/state/jt-recipe-20261014')


def converter():
    path = '/alignment/phase12/hyper-parallel/scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py'
    spec = importlib.util.spec_from_file_location('poststep202_conversion', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model(directory, conversion):
    entries = json.loads((directory/'model/manifest.json').read_text())
    inputs = {row['parameter_name']:np.load(directory/'model'/row['file'],mmap_mode='r') for row in entries}
    return conversion.convert_model(inputs)


def install():
    original = Observer.on_step_end

    def end(self, state, **kwargs):
        original(self, state, **kwargs)
        conversion = converter()
        directory = ROOT/'mf-probe201-r2/step-000201'
        if state.global_step == 201:
            inputs = {}
            for row in json.loads((directory/'optimizer/manifest.json').read_text()):
                if 'muon_m' in row['state']:
                    inputs[row['parameter_name']] = np.load(directory/'optimizer'/row['state']['muon_m']['file'],mmap_mode='r')
            expected = conversion.convert_model(inputs)
            names = {id(value):name for name,value in self.trainer.model.named_parameters()}
            records = []
            for leaf in leaves(self.trainer.optimizer):
                if 'muon' not in type(leaf).__name__.lower():
                    continue
                for group in leaf.param_groups:
                    for parameter in group['params']:
                        name = names[id(parameter)]
                        actual = numpy_value(leaf.state[parameter]['momentum_buffer'])
                        records.append(compare(name, actual, expected[name], expected[name]))
            if len(records) != 28:
                raise ValueError('Expected all28 Muon momentum tensors')
            output = Path('/alignment/phase12/evidence')/(self.output.name+'-momentum201.json')
            output.write_text(json.dumps({'complete':True,'step':201,'records':records},indent=2))
        elif state.global_step == 202:
            before = model(directory,conversion)
            expected = model(ROOT/'mf-grad202-state/step-000202',conversion)
            actual = self.trainer.model.state_dict()
            output = Path('/alignment/phase12/evidence')/(self.output.name+'-poststep202.json')
            records = []
            for name in sorted(expected):
                row = compare(name,numpy_value(actual[name]),expected[name],before[name])
                records.append(row)
                print('[P12-POSTSTEP202]',json.dumps(row),flush=True)
                output.write_text(json.dumps({'complete':False,'records':records},indent=2))
            output.write_text(json.dumps({'complete':True,'step':202,'records':records,
                'scope':'Actual model after independent native update202; reference only compared, never injected.'},indent=2))

    Observer.on_step_end = end
