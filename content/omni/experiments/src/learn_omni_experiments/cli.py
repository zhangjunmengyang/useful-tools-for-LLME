from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, NoReturn

from .core import ResultValidationError, validate_result_payload
from .registry import LESSONS, get_lesson


DEFAULT_OUTPUT_ROOT = Path("artifacts")


def artifact_path(output_root: Path, lesson_id: str) -> Path:
    return output_root / f"lesson{lesson_id}" / "result.json"


def write_result(output_root: Path, lesson_id: str, payload: dict[str, Any]) -> Path:
    lesson = get_lesson(lesson_id)
    validate_result_payload(payload, lesson)
    destination = artifact_path(output_root, lesson.lesson_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def run_one(lesson_id: str, output_root: Path) -> bool:
    try:
        lesson = get_lesson(lesson_id)
        payload = lesson.execute()
        destination = write_result(output_root, lesson.lesson_id, payload)
    except Exception as error:
        print(f"[FAIL] 第 {lesson_id.zfill(2)} 课运行失败")
        print(f"  原因: {type(error).__name__}: {error}")
        return False

    passed = all(payload["checks"].values())
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] 第 {lesson.lesson_id} 课 · {lesson.title}")
    for name, value in payload["checks"].items():
        marker = "✓" if value else "✗"
        print(f"  {marker} {name}")
    print(f"  结果文件: {destination}")
    return passed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON number: {value}")


def _read_result(source: Path) -> Any:
    return json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


def check_one(lesson_id: str, output_root: Path) -> bool:
    try:
        lesson = get_lesson(lesson_id)
    except Exception as error:
        print(f"[FAIL] 无法识别第 {lesson_id} 课")
        print(f"  原因: {type(error).__name__}: {error}")
        return False

    source = artifact_path(output_root, lesson.lesson_id)
    if not source.exists():
        print(
            f"[FAIL] 第 {lesson.lesson_id} 课还没有结果。先运行 "
            f"`python run.py run {lesson.lesson_id}`。",
        )
        return False

    try:
        payload = _read_result(source)
        validate_result_payload(payload, lesson)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ResultValidationError,
        ValueError,
    ) as error:
        print(
            f"[FAIL] 第 {lesson.lesson_id} 课结果文件无效 · {source}",
        )
        print(f"  原因: {type(error).__name__}: {error}")
        return False

    passed = all(payload["checks"].values())
    print(
        f"[{'PASS' if passed else 'FAIL'}] 第 {lesson.lesson_id} 课结果文件"
        f" · {source}",
    )
    if not passed:
        for name, value in payload["checks"].items():
            if not value:
                print(f"  ✗ {name}")
    return passed


def _run_every_lesson(
    action_name: str,
    action: Callable[[str, Path], bool],
    output_root: Path,
) -> bool:
    outcomes: list[bool] = []
    for lesson_id in LESSONS:
        try:
            outcomes.append(bool(action(lesson_id, output_root)))
        except Exception as error:
            outcomes.append(False)
            print(
                f"[FAIL] 第 {lesson_id} 课在 {action_name} 阶段出现未处理异常",
            )
            print(f"  原因: {type(error).__name__}: {error}")
    return all(outcomes)


def run_all(output_root: Path) -> bool:
    return _run_every_lesson("run", run_one, output_root)


def check_all(output_root: Path) -> bool:
    return _run_every_lesson("check", check_one, output_root)


def verify_all(output_root: Path) -> bool:
    run_passed = run_all(output_root)
    check_passed = check_all(output_root)
    return run_passed and check_passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 Learn Omni 的可复现实验",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="结果目录，默认是 experiments/artifacts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="列出 20 课实验")
    list_parser.set_defaults(action="list")

    run_parser = subparsers.add_parser("run", help="运行一课或全部实验")
    run_parser.add_argument("lesson", help="课程编号，例如 01；也可使用 all")
    run_parser.set_defaults(action="run")

    check_parser = subparsers.add_parser("check", help="检查一课的结果文件")
    check_parser.add_argument("lesson", help="课程编号，例如 01；也可使用 all")
    check_parser.set_defaults(action="check")

    verify_parser = subparsers.add_parser(
        "verify-all",
        help="运行并检查全部 20 课",
    )
    verify_parser.set_defaults(action="verify-all")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    output_root = arguments.output_root

    if arguments.action == "list":
        for lesson in LESSONS.values():
            print(f"{lesson.lesson_id}  {lesson.title}")
        return

    if arguments.action == "verify-all":
        passed = verify_all(output_root)
    elif arguments.lesson == "all":
        if arguments.action == "run":
            passed = run_all(output_root)
        else:
            passed = check_all(output_root)
    elif arguments.action == "run":
        passed = run_one(arguments.lesson, output_root)
    else:
        passed = check_one(arguments.lesson, output_root)

    if not passed:
        sys.exit(1)
