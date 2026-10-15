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

"""Independent counterexamples for scalar resource aggregation, without devices."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from all_rank_resources import ALLOCATED, RESERVED, SAMPLES, TIME, TOKENS, _read_rank, main, summarize, write_new_report


def fixture() -> list[dict]:
    """Return alternating slow ranks so max-of-medians gives the wrong answer."""
    return [{step: {'step': step, 'rank': rank, TIME: 9.0 if rank == step - 1 else 1.0,
                   ALLOCATED: 13.0 if rank == 7 else 2.0, RESERVED: 17.0 if rank == 6 else 3.0,
                   TOKENS: 100.0, SAMPLES: 8.0} for step in (1, 2)} for rank in range(8)]


class ResourceTests(unittest.TestCase):
    """Exercise aggregation order, warmup, data duplication and invalid captures."""

    def test_step_max_precedes_median_and_tokens_are_counted_once(self) -> None:
        """Two alternating slow ranks require 18 seconds, not two median times."""
        rows = fixture()
        before = copy.deepcopy(rows)
        result = summarize(rows, 1, 2, 0)
        self.assertEqual(result['median_step_seconds'], 9.0)
        self.assertEqual(result['summed_step_seconds'], 18.0)
        self.assertEqual(result['supervised_tokens'], 200.0)
        self.assertAlmostEqual(result['supervised_tokens_per_second'], 200.0 / 18.0)
        self.assertEqual((result['allocated_gib'], result['reserved_gib']), (13.0, 17.0))
        self.assertEqual(rows, before)
        self.assertFalse(result['precision_acceptance_claimed'])

    def test_warmup_excludes_timing_but_not_cumulative_memory(self) -> None:
        """A warmup peak remains a cumulative peak while its time is excluded."""
        rows = fixture()
        rows[7][1][ALLOCATED] = 30.0
        rows[7][1][TIME] = 99.0
        result = summarize(rows, 1, 2, 1)
        self.assertEqual(result['median_step_seconds'], 9.0)
        self.assertEqual(result['supervised_tokens'], 100.0)
        self.assertEqual(result['allocated_gib'], 30.0)

    def test_missing_rank_step_is_rejected(self) -> None:
        """Incomplete resource observations cannot silently drop the missing rank."""
        rows = fixture()
        rows[7].pop(2)
        with self.assertRaisesRegex(ValueError, 'Missing resource observation'):
            summarize(rows, 1, 2, 0)

    def test_disagreeing_data_count_is_rejected(self) -> None:
        """Cross-rank disagreement must not be hidden by selecting rank0's count."""
        rows = fixture()
        rows[7][2][TOKENS] = 200.0
        with self.assertRaisesRegex(ValueError, 'Logical data count differs'):
            summarize(rows, 1, 2, 0)

    def test_nan_time_is_rejected(self) -> None:
        """Nonfinite input cannot produce a plausible maximum or median."""
        rows = fixture()
        rows[7][2][TIME] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid performance/step_time'):
            summarize(rows, 1, 2, 0)

    def test_boolean_and_fractional_counts_are_rejected(self) -> None:
        """Boolean scalars and fractional logical counts must not look valid."""
        for key, value in ((TOKENS, True), (TOKENS, 1.5), (SAMPLES, 0.5), (TIME, True)):
            with self.subTest(metric=key, value=value):
                rows = fixture()
                for rank_rows in rows:
                    rank_rows[1][key] = value
                with self.assertRaises(ValueError):
                    summarize(rows, 1, 2, 0)

    def test_existing_outputs_are_preserved(self) -> None:
        """An existing precision report cannot be replaced, including a stale precheck race."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'precision.json'
            path.write_bytes(b'original precision evidence')
            with self.assertRaises(FileExistsError):
                write_new_report(path, {'resource_observations': True})
            self.assertEqual(path.read_bytes(), b'original precision evidence')

    def test_cli_directory_alias_cannot_write_receipt(self) -> None:
        """A relative directory alias must not bypass the execution-receipt boundary."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'runs/case'
            run.mkdir(parents=True)
            (root / 'scheduling').mkdir()
            receipt = root / 'scheduling/case.receipt.json'
            alias = run / '..' / 'case'
            with patch('sys.argv', ['resources', str(alias), '1', '200', '--output', str(receipt)]):
                with self.assertRaisesRegex(ValueError, 'execution receipt'):
                    main()
            self.assertFalse(receipt.exists())

    def test_hash_uses_the_parsed_snapshot(self) -> None:
        """Later file changes cannot make the hash describe bytes different from the parsed input."""
        raw = (json.dumps(fixture()[0][1]) + '\n').encode()
        with patch.object(Path, 'read_bytes', side_effect=[raw, b'changed later']) as reader:
            rows, digest = _read_rank(Path('metrics.jsonl'))
        self.assertEqual(rows[1][TOKENS], 100.0)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        reader.assert_called_once()


if __name__ == '__main__':
    unittest.main()
