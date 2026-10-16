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

"""Resource boundaries for the two matching 200-step FP32 dgrad diagnostic arms."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check_pr214_guard import GUARD


class LongDgradGuardTests(unittest.TestCase):
    """Reject expanded or incomplete reservations before launching a container."""

    def setUp(self) -> None:
        """Create an isolated source root for mocked guard decisions."""
        directory = self.enterContext(tempfile.TemporaryDirectory())  # pylint: disable=consider-using-with
        root = Path(directory)
        (root / 'repo').mkdir()
        self.enterContext(patch.object(GUARD, 'ROOT', root))
        self.request = {'host': '129', 'case': 'r129-f637-dgrad-fsdp8-fp32-200-v1', 'cards': 8,
                        'timeout_seconds': 900, 'port': 30378, 'runner': '/work/tools/run_fair_dgrad_200.sh',
                        'source': 'fixed-source', 'patch_sha256': hashlib.sha256(b'').hexdigest(),
                        'files': {}, 'image': sorted(GUARD.ALLOWED_IMAGES)[0]}
        self.idle = {'healthy': True, 'empty': True, 'pids': [], 'core': 0, 'hbm_mb': 3000}

    def test_unpinned_cases_are_rejected_before_external_commands(self) -> None:
        """Neither topology, host, deadline nor port can be silently substituted."""
        for change in ({'cards': 4}, {'host': '125'}, {'timeout_seconds': 840}, {'port': 30379},
                       {'case': 'r129-f637-dgrad-fsdp8-fp32-3'}):
            with self.subTest(change=change), patch.object(GUARD.subprocess, 'check_output') as external:
                with self.assertRaisesRegex(ValueError, 'pinned eight-card 200-step'):
                    GUARD.remote_job({**self.request, **change})
                external.assert_not_called()

    def test_both_arms_require_all_eight_idle_cards(self) -> None:
        """Seven free cards cannot launch either requested eight-card experiment."""
        for layout, port in (('fsdp8', 30378), ('cp2', 30379)):
            request = {**self.request, 'case': f'r129-f637-dgrad-{layout}-fp32-200-v1', 'port': port}
            with self.subTest(layout=layout), \
                    patch.object(GUARD.subprocess, 'check_output', side_effect=['fixed-source\n', b'']), \
                    patch.object(GUARD, 'card_snapshot', return_value={index: self.idle for index in range(7)}):
                receipt = GUARD.remote_job(request)
            self.assertFalse(receipt['attempted'])
            self.assertEqual(receipt['reason'], 'insufficient-free-devices')

    def test_job_appearing_at_recheck_prevents_our_launch(self) -> None:
        """Do not stop another job that starts after the first empty-device snapshot."""
        first = {index: self.idle for index in range(8)}
        second = {**first, 7: {**self.idle, 'empty': False, 'pids': [777]}}
        with patch.object(GUARD.subprocess, 'check_output', side_effect=['fixed-source\n', b'']) as external, \
                patch.object(GUARD, 'card_snapshot', side_effect=[first, second]), \
                patch.object(GUARD.time, 'sleep'), patch.object(GUARD.subprocess, 'run') as execute:
            receipt = GUARD.remote_job(self.request)
        self.assertFalse(receipt['attempted'])
        self.assertEqual(receipt['reason'], 'device-became-busy-before-launch')
        self.assertEqual(external.call_count, 2)
        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
