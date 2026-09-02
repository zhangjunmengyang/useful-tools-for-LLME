from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, NoReturn

from .core import ResultValidationError, validate_result_payload
from .extra.registry import EXTRAS, get_extra
from .extra_core import validate_extra_payload
from .gpu_recipes import RECIPES, get_recipe
from .gpu_smoke import run_smoke
from .hire import run_hire, write_hire
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
        description="运行 Learn CL 的可复现实验",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="结果目录，默认是 experiments/artifacts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="列出 24 课实验")
    list_parser.set_defaults(action="list")

    run_parser = subparsers.add_parser("run", help="运行一课或全部实验")
    run_parser.add_argument("lesson", help="课程编号，例如 01；也可使用 all")
    run_parser.set_defaults(action="run")

    check_parser = subparsers.add_parser("check", help="检查一课的结果文件")
    check_parser.add_argument("lesson", help="课程编号，例如 01；也可使用 all")
    check_parser.set_defaults(action="check")

    verify_parser = subparsers.add_parser(
        "verify-all",
        help="运行并检查全部 24 课",
    )
    verify_parser.set_defaults(action="verify-all")

    capstone_parser = subparsers.add_parser(
        "capstone",
        help="运行 14 日上岗协议（预衡评估，四通道）",
    )
    capstone_parser.set_defaults(action="capstone")

    extra_parser = subparsers.add_parser(
        "extra",
        help="额外 CPU 实验：记忆巩固、自编辑、五日进化",
    )
    extra_parser.add_argument(
        "extra_command",
        choices=("list", "run"),
        help="list 或 run",
    )
    extra_parser.add_argument(
        "name",
        nargs="?",
        default="all",
        help="实验名，或 all",
    )
    extra_parser.set_defaults(action="extra")

    gpu_parser = subparsers.add_parser(
        "gpu",
        help="打印开源仓库 GPU 配方，或跑不下载权重的 smoke",
    )
    gpu_parser.add_argument(
        "gpu_command",
        choices=("list", "print", "smoke"),
        help="list / print / smoke",
    )
    gpu_parser.add_argument("name", nargs="?", default="", help="配方 id")
    gpu_parser.set_defaults(action="gpu")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    output_root = arguments.output_root

    if arguments.action == "list":
        for lesson in LESSONS.values():
            print(f"{lesson.lesson_id}  {lesson.title}")
        print("capstone  北港文具 14 日上岗")
        print("-- extra --")
        for extra in EXTRAS.values():
            print(f"{extra.extra_id}  {extra.title}")
        return

    if arguments.action == "extra":
        _run_extra(arguments, output_root)
        return

    if arguments.action == "gpu":
        _run_gpu(arguments)
        return

    if arguments.action == "capstone":
        payload = run_hire()
        destination = write_hire(output_root)
        passed = all(payload["checks"].values())
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {payload['title']}")
        for name, value in payload["checks"].items():
            marker = "Y" if value else "N"
            print(f"  {marker} {name}")
        print(f"  结果文件: {destination}")
        if not passed:
            sys.exit(1)
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


def extra_artifact_path(output_root: Path, extra_id: str) -> Path:
    return output_root / "extra" / extra_id / "result.json"


def _write_extra(output_root: Path, extra_id: str, payload: dict[str, Any]) -> Path:
    extra = get_extra(extra_id)
    validate_extra_payload(payload, extra)
    destination = extra_artifact_path(output_root, extra.extra_id)
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


def _run_one_extra(extra_id: str, output_root: Path) -> bool:
    try:
        extra = get_extra(extra_id)
        payload = extra.execute()
        destination = _write_extra(output_root, extra.extra_id, payload)
    except Exception as error:
        print(f"[FAIL] extra {extra_id} 运行失败")
        print(f"  原因: {type(error).__name__}: {error}")
        return False
    passed = all(payload["checks"].values())
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] extra {extra.extra_id} · {extra.title}")
    for name, value in payload["checks"].items():
        marker = "Y" if value else "N"
        print(f"  {marker} {name}")
    print(f"  结果文件: {destination}")
    return passed


def _run_extra(arguments: argparse.Namespace, output_root: Path) -> None:
    if arguments.extra_command == "list":
        for extra in EXTRAS.values():
            print(f"{extra.extra_id:10}  {extra.title}  (课 {extra.lesson_hint})")
        return
    names = list(EXTRAS) if arguments.name in ("", "all") else [arguments.name]
    passed = True
    for name in names:
        passed = _run_one_extra(name, output_root) and passed
    if not passed:
        sys.exit(1)


def _run_gpu(arguments: argparse.Namespace) -> None:
    if arguments.gpu_command == "list":
        for recipe in RECIPES:
            print(f"{recipe.recipe_id:16}  课 {recipe.lesson:8}  {recipe.title}")
        return
    if arguments.gpu_command == "smoke":
        payload = run_smoke()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if payload.get("skipped"):
            return
        if payload.get("cuda"):
            print("CUDA 可用。顺序 LoRA smoke 已在 GPU 上跑完。")
        else:
            print("torch 在 CPU 上跑完 smoke。装 CUDA 版 torch 后再上真实仓库。")
        if payload.get("forgot_a") and payload.get("learned_a"):
            return
        if "forgot_a" in payload and not payload["forgot_a"]:
            sys.exit(1)
        return
    if not arguments.name:
        print("gpu print 需要配方 id，先 python3 run.py gpu list")
        sys.exit(2)
    recipe = get_recipe(arguments.name)
    print(f"# {recipe.title}")
    print(f"# 课 {recipe.lesson}")
    print(f"# {recipe.repo}")
    print(f"# 硬件：{recipe.hardware}")
    print(f"# 冒烟：{recipe.smoke}")
    print(f"# {recipe.notes}")
    for command in recipe.commands:
        print(command)

