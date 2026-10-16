#!/usr/bin/env bash
set -eo pipefail
test "$#" = 3
case_name=$1
case "$case_name:$2:$3" in
  r129-a1e6-device-open-v3:30402:1) ;;
  *) exit 2 ;;
esac
output="/work/runs/$case_name"
test ! -e "$output"
mkdir "$output"
record_exit() {
  result=$?
  printf '%s\n' "$result" > "$output/EXIT_CODE"
}
trap record_exit EXIT
source /workspace/env.sh
python_bin=/work/venvs/validation/bin/python
test -x "$python_bin"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd /workspace/hyper-parallel
git rev-parse HEAD > "$output/source-commit.txt"
git status --porcelain=v1 > "$output/source-status.txt"
git diff --binary HEAD > "$output/source.patch"
test "$(cat "$output/source-commit.txt")" = 544526f7db8342c607244fd41009b2a74adb84c8
test ! -s "$output/source-status.txt"
test ! -s "$output/source.patch"
cp /work/tools/probe_a1e6_device_open_v3.py "$output/probe_a1e6_device_open_v3.py"
cp /work/tools/run_fair_a1e6_device_open_v3.sh "$output/run_fair_a1e6_device_open_v3.sh"
set +e
timeout --signal=TERM --kill-after=15s 120s \
  "$python_bin" "$output/probe_a1e6_device_open_v3.py" > "$output/probe.log" 2>&1
probe_result=$?
"$python_bin" - "$output/probe.log" "$output/import-path.json" \
  "$output/production.json" "$output/post-production-diagnostics.json" "$output/selected-device-open.json" <<'PY'
import json
from pathlib import Path
import sys

with Path(sys.argv[1]).open('rb') as stream:
    data = stream.read(1024 * 1024 + 1)
production_records = []
diagnostic_records = []
open_records = []
record_order = []
if len(data) <= 1024 * 1024:
    for line in data.decode('utf-8', errors='replace').splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if 'source_file' in value and 'torch_file' in value and 'scope' in value:
            production_records.append(value)
            record_order.append('production')
        if value.get('record_type') == 'post_production_diagnostics':
            diagnostic_records.append(value)
            record_order.append('diagnostics')
        if value.get('record_type') == 'selected_device_open':
            open_records.append(value)
            record_order.append('device_open')
production = production_records[0] if len(production_records) == 1 else {}
diagnostics = diagnostic_records[0] if len(diagnostic_records) == 1 else {}
device_open = open_records[0] if len(open_records) == 1 else {}
expected = '/workspace/hyper-parallel/hyper_parallel/models/build_options.py'
valid_source = production.get('source_file') == expected
valid_diagnostics = (record_order == ['production', 'diagnostics', 'device_open']
                     and production.get('status') in ('PASS', 'FAIL')
                     and diagnostics.get('production_status') == production.get('status')
                     and isinstance(diagnostics.get('hal_drvGetDevNum'), dict)
                     and diagnostics['hal_drvGetDevNum'].get('status') in ('RETURNED', 'EXCEPTION')
                     and isinstance(diagnostics.get('torch_npu_C_getDeviceCount'), dict)
                     and diagnostics['torch_npu_C_getDeviceCount'].get('status') in ('RETURNED', 'EXCEPTION'))
valid_open = (device_open.get('production_status') == production.get('status')
              and device_open.get('status') in ('NOT_ATTEMPTED', 'OPENED', 'OPENED_NODE_MISMATCH',
                                                'OPEN_ERROR', 'OBSERVATION_ERROR'))
valid = valid_source and valid_diagnostics and valid_open
evidence = {'status': 'PASS' if valid else 'UNVERIFIED',
            'production_record_count': len(production_records), 'diagnostic_record_count': len(diagnostic_records),
            'device_open_record_count': len(open_records),
            'record_order': record_order,
            'source_file': production.get('source_file'), 'torch_file': production.get('torch_file'),
            'source_verified': valid_source, 'diagnostics_verified': valid_diagnostics,
            'device_open_record_verified': valid_open,
            'scope': 'Post-probe log parsing only; no backend imports or device queries'}
for destination, record in zip(sys.argv[2:], (evidence, production, diagnostics, device_open)):
    with Path(destination).open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
raise SystemExit(0 if valid else 1)
PY
import_result=$?
if test "$probe_result" != 0; then
  exit "$probe_result"
fi
exit "$import_result"
