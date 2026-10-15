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


"""Exercise scheduling boundaries without creating containers or using NPU devices."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("guard_under_test", Path(__file__).with_name("fair_resource_guard.py"))
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
RUNNER = "/work/tools/run_fair_pr214_st.sh"
SOURCE = "453618aaad72c3a2f9fb7bf367cd31ca45437d17"
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


class GuardRegressionBoundaryTests(unittest.TestCase):
    """Keep authorization, card selection and cooldown checks in front of launch."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "fixes/pr214-npu-st").mkdir(parents=True)
        self.pin = self.root / "runner.sh"
        self.pin.write_bytes(b"verified runner\n")
        self.request = {
            "host": "129", "case": "pr214-boundary-test", "cards": 4, "timeout_seconds": 900,
            "port": 30357, "runner": RUNNER, "source": SOURCE, "patch_sha256": EMPTY_SHA,
            "files": {"runner.sh": hashlib.sha256(self.pin.read_bytes()).hexdigest()},
            "image": sorted(GUARD.ALLOWED_IMAGES)[0],
        }
        self.root_patch = patch.object(GUARD, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_rejects_wrong_card_counts_before_source_or_device_calls(self):
        for count in (1, 2, 8):
            with self.subTest(count=count), patch.object(GUARD.subprocess, "check_output") as external:
                with self.assertRaisesRegex(ValueError, "exactly four devices"):
                    GUARD.remote_job({**self.request, "cards": count})
                external.assert_not_called()

    def test_unknown_runner_cannot_reach_source_or_devices(self):
        with patch.object(GUARD.subprocess, "check_output") as external:
            with self.assertRaisesRegex(ValueError, "authorized"):
                GUARD.remote_job({**self.request, "runner": "/work/tools/arbitrary.sh"})
            external.assert_not_called()

    def test_unverified_image_cannot_reach_source_or_devices(self):
        with patch.object(GUARD.subprocess, "check_output") as external:
            with self.assertRaisesRegex(ValueError, "immutable image"):
                GUARD.remote_job({**self.request, "image": "latest"})
            external.assert_not_called()

    def test_timeout_cap_is_enforced_before_source_or_devices(self):
        with patch.object(GUARD.subprocess, "check_output") as external:
            with self.assertRaisesRegex(ValueError, "900 seconds"):
                GUARD.remote_job({**self.request, "timeout_seconds": 901})
            external.assert_not_called()

    def test_existing_run_is_not_replaced(self):
        (self.root / "runs" / self.request["case"]).mkdir(parents=True)
        with patch.object(GUARD.subprocess, "check_output") as external:
            with self.assertRaisesRegex(ValueError, "existing run"):
                GUARD.remote_job(self.request)
            external.assert_not_called()

    def test_changed_source_is_rejected_before_device_observation(self):
        with patch.object(GUARD.subprocess, "check_output", side_effect=["different\n", b""]), \
                patch.object(GUARD, "card_snapshot") as observe:
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                GUARD.remote_job(self.request)
            observe.assert_not_called()

    def test_changed_runner_is_rejected_before_device_observation(self):
        self.pin.write_bytes(b"unreviewed content")
        with patch.object(GUARD.subprocess, "check_output", side_effect=[SOURCE + "\n", b""]), \
                patch.object(GUARD, "card_snapshot") as observe:
            with self.assertRaisesRegex(ValueError, "Pinned input changed"):
                GUARD.remote_job(self.request)
            observe.assert_not_called()

    def test_partial_availability_returns_without_creating_container(self):
        idle = {"healthy": True, "empty": True, "pids": [], "core": 0, "hbm_mb": 3000}
        busy = {**idle, "empty": False, "pids": [77]}
        cards = {0: idle, 1: busy, 2: idle, 3: busy}
        with patch.object(GUARD.subprocess, "check_output", side_effect=[SOURCE + "\n", b""]) as external, \
                patch.object(GUARD, "card_snapshot", return_value=cards):
            receipt = GUARD.remote_job(self.request)
        self.assertFalse(receipt["attempted"])
        self.assertTrue(receipt["released"])
        self.assertEqual(receipt["devices"], [0, 2])
        self.assertEqual(receipt["reason"], "insufficient-free-devices")
        self.assertEqual(external.call_count, 2)
        for call in external.call_args_list:
            self.assertEqual(call.args[0][0], "git")
            self.assertEqual(call.args[0][2], str(self.root / "fixes/pr214-npu-st"))

    def test_cooldown_uses_the_more_conservative_clock(self):
        state = {"released_wall": 2000, "released_monotonic": 500}
        self.assertEqual(GUARD.cooldown_remaining(state, 3200, 1700), 0)
        self.assertEqual(GUARD.cooldown_remaining(state, 3199, 1800), 1)
        self.assertEqual(GUARD.cooldown_remaining(state, 4000, 510), 1190)
        self.assertEqual(GUARD.cooldown_remaining(state, 1999, 1700), 1200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
