"""Start the Mechanics Explorer API and frontend dev server."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True, slots=True)
class WorkbenchCommand:
    """描述一个可启动的本地开发进程。"""

    name: str
    command: list[str]
    cwd: Path


def build_commands(
    root: Path,
    *,
    host: str = "127.0.0.1",
    api_port: int = 8766,
    frontend_port: int = 8765,
) -> list[WorkbenchCommand]:
    """构建 API 和前端开发服务器命令。"""
    return [
        WorkbenchCommand(
            name="api",
            command=[
                sys.executable,
                "-m",
                "uvicorn",
                "workbench_api.app:app",
                "--host",
                host,
                "--port",
                str(api_port),
            ],
            cwd=root,
        ),
        WorkbenchCommand(
            name="frontend",
            command=[
                "npm",
                "run",
                "dev",
                "--",
                "--host",
                host,
                "--port",
                str(frontend_port),
            ],
            cwd=root / "frontend",
        ),
    ]


def format_command(command: WorkbenchCommand) -> str:
    """把命令格式化成可复制执行的 shell 片段。"""
    parts = " ".join(shlex.quote(part) for part in command.command)
    return f"{command.name}: cd {shlex.quote(str(command.cwd))} && {parts}"


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并启动本地开发服务。"""
    parser = argparse.ArgumentParser(description="Start the Mechanics Explorer workbench.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to this script's parent repository.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8766)
    parser.add_argument("--frontend-port", type=int, default=8765)
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print commands without starting processes.",
    )
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    commands = build_commands(
        root,
        host=args.host,
        api_port=args.api_port,
        frontend_port=args.frontend_port,
    )

    if args.print:
        for command in commands:
            print(format_command(command))
        return 0

    processes: list[subprocess.Popen[bytes]] = []
    try:
        for command in commands:
            print(format_command(command))
            processes.append(subprocess.Popen(command.command, cwd=command.cwd))
        print(
            f"Workbench: http://{args.host}:{args.frontend_port} "
            f"(API: http://{args.host}:{args.api_port})"
        )
        return_codes = [process.wait() for process in processes]
        return next((code for code in return_codes if code != 0), 0)
    except KeyboardInterrupt:
        return 130
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
