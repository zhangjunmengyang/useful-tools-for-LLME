"""Command line interface for the research toolbox."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

from .capabilities import get_lab_capabilities
from .default_configs import DEFAULT_CONFIGS
from .registry import get_registry


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(prog="python -m workbench_tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available research tools")

    inspect_parser = subparsers.add_parser("inspect", help="Show one tool spec")
    inspect_parser.add_argument("tool_id")

    manifest_parser = subparsers.add_parser("manifest", help="Export toolbox manifest")
    manifest_parser.add_argument(
        "--output",
        help="Optional path to save the manifest JSON",
    )

    sample_parser = subparsers.add_parser("sample-config", help="Print a runnable sample config")
    sample_parser.add_argument("tool_id")

    run_parser = subparsers.add_parser("run", help="Run one research tool")
    run_parser.add_argument("tool_id")
    run_parser.add_argument("--config", required=True, help="Path to JSON input config")
    run_parser.add_argument(
        "--output-dir",
        default="research",
        help="Directory for exported Markdown and JSON artifacts",
    )
    run_parser.add_argument(
        "--no-export",
        action="store_true",
        help="Run without writing Markdown/JSON artifacts",
    )

    batch_parser = subparsers.add_parser("batch", help="Run tools from a JSONL batch")
    batch_parser.add_argument("batch_path", help="JSONL with tool_id and inputs fields")
    batch_parser.add_argument(
        "--output-dir",
        default="research",
        help="Directory for exported Markdown and JSON artifacts",
    )
    batch_parser.add_argument(
        "--no-export",
        action="store_true",
        help="Run without writing Markdown/JSON artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    registry = get_registry()

    if args.command == "list":
        print(json.dumps([spec.to_dict() for spec in registry.list_specs()], ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    if args.command == "inspect":
        print(json.dumps(registry.get_spec(args.tool_id).to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    if args.command == "manifest":
        payload = {
            "tools": [spec.to_dict() for spec in registry.list_specs()],
            "capabilities": [capability.to_dict() for capability in get_lab_capabilities()],
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            Path(args.output).write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return 0

    if args.command == "sample-config":
        registry.get_spec(args.tool_id)
        payload = DEFAULT_CONFIGS.get(args.tool_id, {})
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    if args.command == "run":
        config_path = Path(args.config)
        inputs = json.loads(config_path.read_text(encoding="utf-8"))
        run = registry.run(
            args.tool_id,
            inputs,
            export=not args.no_export,
            output_dir=args.output_dir,
        )
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if run.status == "success" else 1

    if args.command == "batch":
        batch_path = Path(args.batch_path)
        runs = []
        for line_number, line in enumerate(batch_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            tool_id = item.get("tool_id")
            inputs = item.get("inputs", {})
            if not tool_id:
                runs.append(
                    {
                        "line": line_number,
                        "status": "error",
                        "error": "`tool_id` is required",
                    }
                )
                continue
            run = registry.run(
                tool_id,
                inputs,
                export=not args.no_export,
                output_dir=args.output_dir,
            )
            runs.append(run.to_dict())
        has_error = any(run.get("status") == "error" for run in runs)
        summary = {
            "total": len(runs),
            "success": sum(1 for run in runs if run.get("status") == "success"),
            "error": sum(1 for run in runs if run.get("status") == "error"),
        }
        print(json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if has_error else 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
