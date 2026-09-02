"""Split lesson markdown into read / learn / play slices."""

from __future__ import annotations

import re
from typing import Any

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.M)
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.S)

LEARN_HINTS = (
    "问题",
    "准备",
    "练习",
    "检查",
    "验收",
    "做什么",
    "记住",
    "误区",
    "gates",
    "principles",
    "outcome",
    "学",
    "作业",
    "目标",
    "learn",
    "check yourself",
)
PLAY_HINTS = (
    "实验",
    "动手",
    "lab",
    "跑通",
    "命令",
    "playground",
    "玩",
    "操作",
    "配方",
    "recipe",
    "play",
    "try it",
)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return YAML-like frontmatter and the remaining body."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw = match.group(1)
    meta: dict[str, Any] = {}
    current_list: str | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if current_list and (line.startswith("  - ") or line.startswith("- ")):
            value = line.split("- ", 1)[1].strip().strip("\"'")
            meta.setdefault(current_list, []).append(value)
            continue
        keyed = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", line)
        if not keyed:
            continue
        key, value = keyed.group(1), keyed.group(2).strip()
        if value == "" or value == "[]":
            current_list = key
            meta[key] = []
            continue
        current_list = None
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            meta[key] = [part.strip().strip("\"'") for part in inner.split(",") if part.strip()]
        else:
            meta[key] = value.strip("\"'")
    return meta, text[match.end() :]


def split_sections(markdown: str) -> list[dict[str, str]]:
    """Split markdown into heading-bounded sections."""
    matches = list(HEADING_RE.finditer(markdown))
    if not matches:
        title = first_title(markdown)
        return [{"level": "1", "title": title, "body": markdown.strip()}]

    sections: list[dict[str, str]] = []
    preface = markdown[: matches[0].start()].strip()
    if preface:
        sections.append({"level": "0", "title": "导读", "body": preface})
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections.append(
            {
                "level": str(len(match.group(1))),
                "title": match.group(2).strip(),
                "body": markdown[match.end() : end].strip(),
            }
        )
    return sections


def first_title(markdown: str) -> str:
    """Return the first heading, or a fallback."""
    match = HEADING_RE.search(markdown)
    if match:
        return match.group(2).strip()
    return "未命名课"


def _hinted(title: str, hints: tuple[str, ...]) -> bool:
    lowered = title.lower()
    return any(hint.lower() in lowered for hint in hints)


def learn_sections(sections: list[dict[str, str]]) -> list[dict[str, str]]:
    """Pick sections that teach, check, or set a task."""
    picked = [section for section in sections if _hinted(section["title"], LEARN_HINTS)]
    if picked:
        return picked
    return [section for section in sections if section["level"] in {"1", "2"}][:3]


def play_sections(sections: list[dict[str, str]]) -> list[dict[str, str]]:
    """Pick sections that ask the reader to run something."""
    picked = [section for section in sections if _hinted(section["title"], PLAY_HINTS)]
    if picked:
        return picked
    code_blocks = [section for section in sections if "```" in section["body"]]
    return code_blocks[:4]


def render_sections(sections: list[dict[str, str]]) -> str:
    """Join sections back into markdown."""
    chunks: list[str] = []
    for section in sections:
        if section["level"] == "0":
            chunks.append(section["body"])
            continue
        hashes = "#" * max(int(section["level"]), 1)
        chunks.append(f"{hashes} {section['title']}\n\n{section['body']}".rstrip())
    return "\n\n".join(chunk for chunk in chunks if chunk).strip()
