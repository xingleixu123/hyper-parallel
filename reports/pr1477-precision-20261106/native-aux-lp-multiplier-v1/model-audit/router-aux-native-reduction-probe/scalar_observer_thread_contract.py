"""Actual aux state capture survives a backward thread with no ContextVar."""
import hashlib
import json
from pathlib import Path
import sys
import threading

SOURCE = Path('/home/xxl/codes-github/.codex-tmp/pr1477-20261106-fourcard-source-v3')
sys.path.insert(0, str(SOURCE))
import torch
from hyper_parallel.components.functional import aux_loss as aux

assert Path(aux.__file__).is_relative_to(SOURCE)
observed = {}
parameter = torch.tensor(2., requires_grad=True)
with aux.aux_loss_scale_context():
    output = aux.aux_loss_auto_scale(parameter * 0, parameter.square())
    foundation = output.sum() * 0
    aux.bind_aux_loss_scale(foundation)
    state_object = aux._AUX_LOSS_STATE.get()
    def observer(gradient):
        observed['scalar_gradient'] = float(gradient)
        observed['saved_state_scale'] = float(state_object.scale)
        observed['thread_context_is_None'] = aux._AUX_LOSS_STATE.get() is None
    handle = foundation.register_hook(observer)
    def backward():
        (foundation * 7).backward()
    thread = threading.Thread(target=backward)
    thread.start()
    thread.join(15)
    assert not thread.is_alive()
    handle.remove()
assert observed == {'scalar_gradient': 7., 'saved_state_scale': 7., 'thread_context_is_None': True}
assert float(parameter.grad) == 28.
result = {'status': 'actual_source_aux_scale_cross_thread_closure_contract_PASS',
    'actual_source': aux.__file__, 'source_sha256': hashlib.sha256(Path(aux.__file__).read_bytes()).hexdigest(),
    'observed': observed, 'native_aux_gradient': float(parameter.grad), 'expected_aux_gradient': 28.,
    'NPU_executed': False,
    'scope': 'One actual aux context/bind/auto-scale CPU graph; backward in a new Python thread with ContextVar absent. No mocks, product changes or gradient compensation.'}
(Path(__file__).resolve().parent / 'scalar-observer-thread-contract.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
