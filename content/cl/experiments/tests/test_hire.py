from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from learn_cl_experiments.hire import run_channel, run_hire, write_hire  # noqa: E402


class HireProtocolTest(unittest.TestCase):
    def test_all_checks_pass(self) -> None:
        payload = run_hire()
        failed = [name for name, value in payload["checks"].items() if not value]
        self.assertEqual(failed, [], msg=str(payload["checks"]))

    def test_channels_split_like_lesson_16_matrix(self) -> None:
        frozen = run_channel("frozen")
        rag = run_channel("rag")
        memory = run_channel("memory_skill")
        full = run_channel("full")
        self.assertFalse(frozen["seat_probe"])
        self.assertFalse(rag["seat_probe"])
        self.assertTrue(memory["seat_probe"])
        self.assertTrue(full["seat_probe"])
        self.assertFalse(memory["rule_ok"])
        self.assertTrue(full["rule_ok"])
        self.assertEqual(frozen["skill_count"], 0)
        self.assertGreaterEqual(memory["skill_count"], 1)

    def test_write_hire_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_hire(Path(directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"]["name"], "learn-cl-hire-result")
        self.assertTrue(all(payload["checks"].values()))


if __name__ == "__main__":
    unittest.main()
