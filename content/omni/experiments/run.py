from __future__ import annotations

import sys
from pathlib import Path


if sys.version_info < (3, 10):
    raise SystemExit(
        "Learn Omni 实验需要 Python 3.10 或更高版本。"
        f"当前版本是 {sys.version_info.major}.{sys.version_info.minor}。"
        "请先执行 `python3 --version`，再切换到符合要求的解释器。",
    )

SOURCE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from learn_omni_experiments.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
