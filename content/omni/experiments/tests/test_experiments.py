from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from learn_omni_experiments import cli  # noqa: E402
from learn_omni_experiments.cli import (  # noqa: E402
    artifact_path,
    check_all,
    check_one,
    run_all,
    run_one,
    verify_all,
    write_result,
)
from learn_omni_experiments.core import (  # noqa: E402
    RESULT_FIELDS,
    RESULT_SCHEMA,
    ResultValidationError,
    runtime_metadata,
)
from learn_omni_experiments.registry import LESSONS  # noqa: E402


class ExperimentContractTest(unittest.TestCase):
    def test_all_registered_lessons_run_and_write_valid_artifacts(self) -> None:
        self.assertEqual(list(LESSONS), [f"{index:02d}" for index in range(1, 61)])
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            for lesson_id in LESSONS:
                with self.subTest(lesson_id=lesson_id):
                    self.assertTrue(run_one(lesson_id, output_root))
                    self.assertTrue(check_one(lesson_id, output_root))
                    payload = json.loads(
                        artifact_path(output_root, lesson_id).read_text(
                            encoding="utf-8",
                        ),
                    )
                    self.assertEqual(payload["lesson_id"], lesson_id)
                    self.assertEqual(set(payload), RESULT_FIELDS)
                    self.assertEqual(payload["schema"], RESULT_SCHEMA)
                    self.assertEqual(payload["runtime"], runtime_metadata())
                    self.assertEqual(
                        payload["source_digest"],
                        LESSONS[lesson_id].source_digest(),
                    )
                    self.assertIsInstance(payload["summary"], str)
                    self.assertTrue(payload["summary"].strip())
                    self.assertIsInstance(payload["metrics"], dict)
                    self.assertTrue(payload["metrics"])
                    self.assertTrue(all(payload["checks"].values()))
                    replayed = LESSONS[lesson_id].execute()
                    self.assertEqual(payload, replayed)
                    json.dumps(payload, allow_nan=False)

    def test_run_and_check_all_visit_every_lesson_after_failure(self) -> None:
        lesson_ids = {"01": object(), "02": object(), "03": object()}

        for command_name, command in (
            ("run_one", run_all),
            ("check_one", check_all),
        ):
            calls: list[str] = []

            def action(lesson_id: str, output_root: Path) -> bool:
                calls.append(lesson_id)
                if lesson_id == "01":
                    return False
                if lesson_id == "02":
                    raise RuntimeError("planned test failure")
                return True

            with self.subTest(command=command_name):
                with (
                    mock.patch.object(cli, "LESSONS", lesson_ids),
                    mock.patch.object(cli, command_name, side_effect=action),
                    redirect_stdout(io.StringIO()) as output,
                ):
                    self.assertFalse(command(Path("unused")))
                self.assertEqual(calls, ["01", "02", "03"])
                self.assertIn("planned test failure", output.getvalue())

    def test_verify_all_checks_every_artifact_even_when_run_fails(self) -> None:
        with (
            mock.patch.object(cli, "run_all", return_value=False) as run_mock,
            mock.patch.object(cli, "check_all", return_value=True) as check_mock,
        ):
            self.assertFalse(verify_all(Path("unused")))
        run_mock.assert_called_once_with(Path("unused"))
        check_mock.assert_called_once_with(Path("unused"))

    def test_run_one_turns_exceptions_into_a_friendly_failure(self) -> None:
        with (
            mock.patch.object(
                cli,
                "get_lesson",
                side_effect=RuntimeError("broken experiment"),
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertFalse(run_one("01", Path("unused")))
        self.assertIn("[FAIL]", output.getvalue())
        self.assertIn("broken experiment", output.getvalue())

    def test_check_rejects_malformed_and_non_object_json(self) -> None:
        invalid_documents = (
            "{",
            "[]",
            '{"lesson_id":"01","lesson_id":"01"}',
            '{"value":NaN}',
        )
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            source = artifact_path(output_root, "01")
            source.parent.mkdir(parents=True)
            for document in invalid_documents:
                with self.subTest(document=document):
                    source.write_text(document, encoding="utf-8")
                    with redirect_stdout(io.StringIO()) as output:
                        self.assertFalse(check_one("01", output_root))
                    self.assertIn("[FAIL]", output.getvalue())
                    self.assertIn(str(source), output.getvalue())

    def test_check_rejects_missing_stale_and_forged_fields(self) -> None:
        lesson = LESSONS["01"]
        valid_payload = lesson.execute()

        tampered_payloads: dict[str, dict[str, object]] = {}
        for field in RESULT_FIELDS:
            missing = copy.deepcopy(valid_payload)
            del missing[field]
            tampered_payloads[f"missing_{field}"] = missing

        extra = copy.deepcopy(valid_payload)
        extra["unregistered"] = True
        tampered_payloads["unexpected_field"] = extra

        wrong_title = copy.deepcopy(valid_payload)
        wrong_title["title"] = "伪造标题"
        tampered_payloads["wrong_title"] = wrong_title

        blank_summary = copy.deepcopy(valid_payload)
        blank_summary["summary"] = " "
        tampered_payloads["blank_summary"] = blank_summary

        empty_metrics = copy.deepcopy(valid_payload)
        empty_metrics["metrics"] = {}
        tampered_payloads["empty_metrics"] = empty_metrics

        integer_check = copy.deepcopy(valid_payload)
        first_check = next(iter(integer_check["checks"]))
        integer_check["checks"][first_check] = 1
        tampered_payloads["integer_check"] = integer_check

        stale_source = copy.deepcopy(valid_payload)
        stale_source["source_digest"]["value"] = "0" * 64
        tampered_payloads["stale_source"] = stale_source

        forged_runtime = copy.deepcopy(valid_payload)
        forged_runtime["runtime"]["python_version"] = "0.0.0"
        tampered_payloads["forged_runtime"] = forged_runtime

        wrong_schema = copy.deepcopy(valid_payload)
        wrong_schema["schema"]["version"] = 999
        tampered_payloads["wrong_schema"] = wrong_schema

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            source = artifact_path(output_root, "01")
            source.parent.mkdir(parents=True)
            for name, payload in tampered_payloads.items():
                with self.subTest(tampering=name):
                    source.write_text(
                        json.dumps(payload, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    with redirect_stdout(io.StringIO()):
                        self.assertFalse(check_one("01", output_root))

    def test_write_uses_the_same_strict_validator_as_check(self) -> None:
        payload = LESSONS["01"].execute()
        payload["source_digest"]["value"] = "f" * 64
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with self.assertRaises(ResultValidationError):
                write_result(output_root, "01", payload)
            self.assertFalse(artifact_path(output_root, "01").exists())

    def test_valid_artifact_with_a_failed_check_does_not_pass(self) -> None:
        payload = LESSONS["01"].execute()
        first_check = next(iter(payload["checks"]))
        payload["checks"][first_check] = False
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            write_result(output_root, "01", payload)
            with redirect_stdout(io.StringIO()) as output:
                self.assertFalse(check_one("01", output_root))
            self.assertIn(first_check, output.getvalue())


if __name__ == "__main__":
    unittest.main()
