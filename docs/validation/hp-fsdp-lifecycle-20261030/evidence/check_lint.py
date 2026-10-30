"""Compare full-file lint with the pinned base without editing the source tree."""
import collections
import json
from pathlib import Path
import subprocess
import sys

task = Path(__file__).resolve().parent
repo = task / 'repository'
base = 'e0cdf62c9d66f1f9ec48c6f15df87dcf4ce80a5f'
changed = subprocess.check_output(['git', 'diff', base, '--name-only'], cwd=repo, text=True).splitlines()
added = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=repo, text=True).splitlines()
paths = [p for p in changed + added if p.endswith('.py')]
tracked = [p for p in paths if subprocess.run(['git', 'cat-file', '-e', f'{base}:{p}'], cwd=repo,
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0]
baseline = task / 'lint-baseline'
for relative in tracked:
    target = baseline / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(subprocess.check_output(['git', 'show', f'{base}:{relative}'], cwd=repo))

def lint(targets, label):
    proc = subprocess.run([sys.executable, '-m', 'pylint', '--rcfile=.pylintrc', '--output-format=json', *targets],
                          cwd=repo, text=True, capture_output=True, check=False)
    (task / f'pylint-{label}.json').write_text(proc.stdout)
    (task / f'pylint-{label}.stderr').write_text(proc.stderr)
    return json.loads(proc.stdout)

current = lint(paths, 'current')
previous = lint([str(baseline / p) for p in tracked], 'base')
def key(item):
    return Path(item['path']).name, item['message-id'], item['symbol'], item['obj'], item['message']
old = collections.Counter(key(m) for m in previous)
new = []
for message in current:
    identity = key(message)
    if old[identity]:
        old[identity] -= 1
    else:
        new.append(message)
result = {'base_findings': len(previous), 'current_findings': len(current), 'new_findings': new}
(task / 'lint-comparison.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
sys.exit(bool(new))
