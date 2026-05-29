"""Development launcher tests for Mechanics Explorer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import io
import unittest

from scripts.dev_workbench import build_commands, main


class DevWorkbenchTest(unittest.TestCase):
    """验证一键启动脚本保持可审计、可打印。"""

    def test_build_commands_uses_api_and_frontend_defaults(self):
        root = Path("/tmp/workbench")

        commands = build_commands(root)

        self.assertEqual(len(commands), 2)
        api_command, frontend_command = commands
        self.assertEqual(api_command.name, "api")
        self.assertEqual(api_command.cwd, root)
        self.assertIn("uvicorn", api_command.command)
        self.assertIn("workbench_api.app:app", api_command.command)
        self.assertIn("8001", api_command.command)
        self.assertEqual(frontend_command.name, "frontend")
        self.assertEqual(frontend_command.cwd, root / "frontend")
        self.assertEqual(frontend_command.command[:3], ["npm", "run", "dev"])
        self.assertIn("5173", frontend_command.command)

    def test_main_print_mode_does_not_launch_processes(self):
        with patch("scripts.dev_workbench.subprocess.Popen") as popen, patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            exit_code = main(["--root", "/tmp/workbench", "--print"])

        self.assertEqual(exit_code, 0)
        self.assertFalse(popen.called)
        output = stdout.getvalue()
        self.assertIn("api:", output)
        self.assertIn("frontend:", output)
        self.assertIn("workbench_api.app:app", output)
        self.assertIn("npm run dev", output)


if __name__ == "__main__":
    unittest.main()
