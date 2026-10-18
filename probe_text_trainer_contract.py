"""Reproduce the optional subclass initialization gap without loading a backend.

This executes the exact constructor AST from the fixed reviewed Git commit.
It is an audit probe, not production code or a training patch.
"""
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
WORKTREE = Path(r'C:\Users\90803\.codex\worktrees\jt-mf-archive-profile\hyper-parallel')
COMMIT = '4ccb63a84304b4d968fb483f5d86b4a385be23df'
SOURCE_PATH = 'hyper_parallel/trainer/text_trainer.py'
source = subprocess.check_output(['git', '-C', str(WORKTREE), 'show', f'{COMMIT}:{SOURCE_PATH}'], text=True)
tree = ast.parse(source)
definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TextTrainer')
constructor = next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')


class Base:
    """Only the subtype relationship is relevant before the failing setup."""


class Specialized(Base):
    def __init__(self, config):
        self.config = config
        self.initialized_stage_state = 'ready'

    def _setup(self):
        return self.initialized_stage_state


module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                         constructor], type_ignores=[])
ast.fix_missing_locations(module)
namespace = {'BaseTrainer': Base}
exec(compile(module, f'{COMMIT}:{SOURCE_PATH}', 'exec'), namespace)
surrogate = type('ExactConstructorProbe', (), {'__init__': namespace['__init__']})
try:
    surrogate(SimpleNamespace(), base_trainer_class=Specialized)
except AttributeError as error:
    failure = str(error)
else:
    raise AssertionError('Expected the documented initialization gap')
normal = Specialized(SimpleNamespace())._setup()
assert normal == 'ready'
report = {'commit': COMMIT, 'source': SOURCE_PATH, 'mode': 'exact_constructor_ast_no_backend',
          'text_trainer_subclass_result': failure, 'normal_subclass_result': normal,
          'reproduced': True, 'training_or_device_used': False}
(ROOT / 'text-trainer-contract-result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
