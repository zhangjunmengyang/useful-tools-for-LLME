"""Topic registry: four first-class courses, sibling markdown or local LLM."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from learn_platform.markdown_split import (
    first_title,
    learn_sections,
    parse_frontmatter,
    play_sections,
    render_sections,
    split_sections,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
TOPICS_PATH = REPO_ROOT / "content" / "topics.json"

UNIT_RE = re.compile(
    r'id:\s*"(?P<id>[\w-]+)"\s*,\s*order:\s*(?P<order>\d+)\s*,\s*title:\s*"(?P<title>[^"]+)"\s*,\s*question:\s*"(?P<question>[^"]*)"',
)
LESSON_HEAD_RE = re.compile(
    r'\{\s*id:\s*"(?P<id>\d+)"\s*,\s*slug:\s*"(?P<slug>[^"]+)"\s*,\s*shortTitle:\s*"(?P<title>[^"]+)"\s*,\s*unit:\s*unitById(?:\.(?P<unit_dot>[\w-]+)|\[\s*"(?P<unit_br>[\w-]+)"\s*\])',
    re.S,
)
QUESTION_RE = re.compile(r'essentialQuestion:\s*"((?:\\.|[^"\\])*)"')


def _load_topics_file() -> list[dict[str, Any]]:
    payload = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
    return list(payload["topics"])


def _resolve_sibling_root(spec: dict[str, Any]) -> Path | None:
    for relative in spec.get("project_roots", []):
        candidate = (PROJECT_ROOT / relative).resolve()
        lessons = candidate / spec["lessons_dir"]
        if lessons.is_dir():
            return candidate
    return None


def _local_root(spec: dict[str, Any]) -> Path:
    return (REPO_ROOT / spec["root"]).resolve()


def list_topics() -> list[dict[str, Any]]:
    """Return switcher entries. Missing sibling content stays listed, with a reason."""
    topics: list[dict[str, Any]] = []
    for spec in _load_topics_file():
        ready, source, note = _topic_status(spec)
        topics.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "title_en": spec.get("title_en") or spec["title"],
                "short": spec["short"],
                "short_en": spec.get("short_en") or spec["short"],
                "blurb": spec["blurb"],
                "blurb_en": spec.get("blurb_en") or spec["blurb"],
                "kind": spec["kind"],
                "ready": ready,
                "source": source,
                "note": note,
                "modes": ["read", "learn", "play"],
            }
        )
    return topics


def _topic_status(spec: dict[str, Any]) -> tuple[bool, str, str]:
    if spec["kind"] == "local_markdown":
        root = _local_root(spec)
        lessons = root / "lessons"
        if lessons.is_dir() and any(lessons.glob("*.md")):
            return True, str(root), ""
        return False, str(root), "本仓库还没有 LLM 课文件。"
    root = _resolve_sibling_root(spec)
    if root is None:
        tried = ", ".join(spec.get("project_roots", []))
        return False, "", f"找不到课程正文。试过：{tried}"
    return True, str(root), ""


def get_spec(topic_id: str) -> dict[str, Any]:
    for spec in _load_topics_file():
        if spec["id"] == topic_id:
            return spec
    raise KeyError(topic_id)


def topic_outline(topic_id: str) -> dict[str, Any]:
    spec = get_spec(topic_id)
    if spec["kind"] == "local_markdown":
        return _local_outline(spec)
    return _sibling_outline(spec)


def topic_lesson(topic_id: str, lesson_id: str) -> dict[str, Any]:
    spec = get_spec(topic_id)
    if spec["kind"] == "local_markdown":
        return _local_lesson(spec, lesson_id)
    return _sibling_lesson(spec, lesson_id)


def _local_outline(spec: dict[str, Any]) -> dict[str, Any]:
    root = _local_root(spec)
    course = json.loads((root / "course.json").read_text(encoding="utf-8"))
    units = {unit["id"]: {**unit, "lessons": []} for unit in course["units"]}
    lessons: list[dict[str, Any]] = []
    for path in sorted((root / "lessons").glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        lesson_id = str(meta.get("id") or path.stem)
        unit_id = str(meta.get("unit") or "input")
        number_match = re.match(r"^(\d+)", path.stem)
        en_meta = _english_lesson_meta(root, path, lesson_id)
        item = {
            "id": lesson_id,
            "title": str(meta.get("title") or first_title(body)),
            "title_en": str(en_meta.get("title") or meta.get("title") or first_title(body)),
            "summary": str(meta.get("summary") or ""),
            "summary_en": str(en_meta.get("summary") or meta.get("summary") or ""),
            "unit_id": unit_id,
            "number": number_match.group(1) if number_match else None,
            "play_tools": list(meta.get("play_tools") or []),
        }
        lessons.append(item)
        units.setdefault(unit_id, {"id": unit_id, "title": unit_id, "question": "", "lessons": []})
        units[unit_id]["lessons"].append(item)
    for unit in course["units"]:
        if unit["id"] in units:
            units[unit["id"]]["title_en"] = unit.get("title_en") or unit["title"]
            units[unit["id"]]["question_en"] = unit.get("question_en") or unit.get("question") or ""
    unit_list = [units[unit["id"]] for unit in course["units"] if units[unit["id"]]["lessons"]]
    extra = [unit for key, unit in units.items() if key not in {u["id"] for u in course["units"]} and unit["lessons"]]
    return {
        "id": spec["id"],
        "title": spec["title"],
        "title_en": spec.get("title_en") or spec["title"],
        "blurb": spec["blurb"],
        "blurb_en": spec.get("blurb_en") or spec["blurb"],
        "summary": course.get("summary", spec["blurb"]),
        "summary_en": course.get("summary_en") or course.get("summary", spec.get("blurb_en") or spec["blurb"]),
        "ready": True,
        "source": str(root),
        "original_url": None,
        "units": unit_list + extra,
        "lessons": lessons,
        "default_lesson_id": lessons[0]["id"] if lessons else None,
    }


def _local_lesson(spec: dict[str, Any], lesson_id: str) -> dict[str, Any]:
    root = _local_root(spec)
    match: Path | None = None
    for path in (root / "lessons").glob("*.md"):
        meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        if str(meta.get("id") or path.stem) == lesson_id:
            match = path
            break
    if match is None:
        raise KeyError(lesson_id)
    raw = match.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)
    sections = split_sections(body)
    play_tools = list(meta.get("play_tools") or [])
    checkpoints = list(meta.get("checkpoints") or [])
    learn_md = render_sections(learn_sections(sections))
    if checkpoints:
        checklist = "\n".join(f"- {item}" for item in checkpoints)
        learn_md = (learn_md + "\n\n## 学完能说清什么\n\n" + checklist).strip()
    play_md = render_sections(play_sections(sections))
    english = _english_lesson_payload(root, match, lesson_id)
    return {
        "id": lesson_id,
        "topic_id": spec["id"],
        "title": str(meta.get("title") or first_title(body)),
        "title_en": english["title"] if english else str(meta.get("title") or first_title(body)),
        "summary": str(meta.get("summary") or ""),
        "summary_en": english["summary"] if english else str(meta.get("summary") or ""),
        "unit_id": str(meta.get("unit") or ""),
        "format": "markdown",
        "read": body.strip(),
        "learn": learn_md,
        "play": play_md,
        "read_en": english["read"] if english else None,
        "learn_en": english["learn"] if english else None,
        "play_en": english["play"] if english else None,
        "play_tools": play_tools,
        "checkpoints": checkpoints,
        "body_locale": "both" if english else "zh",
        "original_url": None,
        "source_path": str(match),
    }


def _english_lesson_path(root: Path, zh_path: Path, lesson_id: str) -> Path | None:
    en_dir = root / "lessons-en"
    if not en_dir.is_dir():
        return None
    same_name = en_dir / zh_path.name
    if same_name.is_file():
        return same_name
    for path in en_dir.glob("*.md"):
        meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        if str(meta.get("id") or path.stem) == lesson_id:
            return path
    return None


def _english_lesson_meta(root: Path, zh_path: Path, lesson_id: str) -> dict[str, Any]:
    path = _english_lesson_path(root, zh_path, lesson_id)
    if path is None:
        return {}
    meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
    return meta


def _english_lesson_payload(root: Path, zh_path: Path, lesson_id: str) -> dict[str, str] | None:
    path = _english_lesson_path(root, zh_path, lesson_id)
    if path is None:
        return None
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)
    sections = split_sections(body)
    learn_md = render_sections(learn_sections(sections))
    checkpoints = list(meta.get("checkpoints") or [])
    if checkpoints:
        checklist = "\n".join(f"- {item}" for item in checkpoints)
        learn_md = (learn_md + "\n\n## What you should be able to say\n\n" + checklist).strip()
    return {
        "title": str(meta.get("title") or first_title(body)),
        "summary": str(meta.get("summary") or ""),
        "read": body.strip(),
        "learn": learn_md,
        "play": render_sections(play_sections(sections)),
    }


def _read_course_data(root: Path, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    meta_path = root / spec["meta_file"]
    if not meta_path.is_file():
        return [], {}
    text = meta_path.read_text(encoding="utf-8")
    units_chunk = text
    if "export const courseUnits" in text:
        units_chunk = text.split("export const courseUnits", 1)[1].split("] as const", 1)[0]
    units = [
        {
            "id": match.group("id"),
            "order": int(match.group("order")),
            "title": match.group("title"),
            "question": match.group("question"),
            "lessons": [],
        }
        for match in UNIT_RE.finditer(units_chunk)
    ]
    lessons_by_num: dict[str, dict[str, Any]] = {}
    for match in LESSON_HEAD_RE.finditer(text):
        block_start = match.start()
        block_end = text.find("\n  },", block_start)
        block = text[block_start : block_end if block_end != -1 else block_start + 1200]
        question_match = QUESTION_RE.search(block)
        lessons_by_num[match.group("id")] = {
            "number": match.group("id"),
            "slug": match.group("slug"),
            "title": match.group("title"),
            "unit_id": match.group("unit_dot") or match.group("unit_br"),
            "question": question_match.group(1).replace('\\"', '"') if question_match else "",
        }
    return units, lessons_by_num


def _lesson_file_number(path: Path) -> str:
    match = re.match(r"^(\d+)", path.stem)
    return match.group(1) if match else path.stem


def _sibling_outline(spec: dict[str, Any]) -> dict[str, Any]:
    root = _resolve_sibling_root(spec)
    if root is None:
        ready, source, note = _topic_status(spec)
        return {
            "id": spec["id"],
            "title": spec["title"],
            "title_en": spec.get("title_en") or spec["title"],
            "blurb": spec["blurb"],
            "blurb_en": spec.get("blurb_en") or spec["blurb"],
            "summary": spec["blurb"],
            "ready": ready,
            "source": source,
            "note": note,
            "original_url": spec.get("original_base"),
            "units": [],
            "lessons": [],
            "default_lesson_id": None,
        }
    units, by_number = _read_course_data(root, spec)
    unit_map = {unit["id"]: unit for unit in units}
    lessons_dir = root / spec["lessons_dir"]
    lessons: list[dict[str, Any]] = []
    for path in sorted(lessons_dir.glob("*.md")):
        number = _lesson_file_number(path)
        meta = by_number.get(number, {})
        body = path.read_text(encoding="utf-8")
        title = meta.get("title") or first_title(body)
        unit_id = meta.get("unit_id") or "other"
        item = {
            "id": path.stem,
            "title": title,
            "summary": meta.get("question") or "",
            "unit_id": unit_id,
            "number": number,
            "slug": meta.get("slug") or path.stem,
            "play_tools": [],
        }
        lessons.append(item)
        if unit_id not in unit_map:
            unit_map[unit_id] = {
                "id": unit_id,
                "order": 99,
                "title": "其他",
                "question": "",
                "lessons": [],
            }
            units.append(unit_map[unit_id])
        unit_map[unit_id]["lessons"].append(item)
    units = [unit for unit in units if unit["lessons"]]
    return {
        "id": spec["id"],
        "title": spec["title"],
        "title_en": spec.get("title_en") or spec["title"],
        "blurb": spec["blurb"],
        "blurb_en": spec.get("blurb_en") or spec["blurb"],
        "summary": spec["blurb"],
        "ready": True,
        "source": str(root),
        "original_url": spec.get("original_base"),
        "units": units,
        "lessons": lessons,
        "default_lesson_id": lessons[0]["id"] if lessons else None,
    }


def _sibling_lesson(spec: dict[str, Any], lesson_id: str) -> dict[str, Any]:
    root = _resolve_sibling_root(spec)
    if root is None:
        raise KeyError(lesson_id)
    path = root / spec["lessons_dir"] / f"{lesson_id}.md"
    if not path.is_file():
        matches = list((root / spec["lessons_dir"]).glob(f"{lesson_id}*.md"))
        if not matches:
            raise KeyError(lesson_id)
        path = matches[0]
    body = path.read_text(encoding="utf-8")
    _, by_number = _read_course_data(root, spec)
    number = _lesson_file_number(path)
    meta = by_number.get(number, {})
    sections = split_sections(body)
    learn_md = render_sections(learn_sections(sections))
    question = meta.get("question") or ""
    if question and question not in learn_md:
        learn_md = f"## 这一课要回答的问题\n\n{question}\n\n{learn_md}".strip()
    play_md = render_sections(play_sections(sections))
    slug = meta.get("slug") or ""
    original = None
    if spec.get("original_base") and slug:
        original = f"{spec['original_base'].rstrip('/')}{spec.get('original_course_prefix', '/course')}/{slug}"
    if not play_md:
        play_md = (
            "这门课的动手部分在原站实验页和课内命令里。"
            + (f" 原课地址：{original}" if original else "")
        )
    return {
        "id": path.stem,
        "topic_id": spec["id"],
        "title": meta.get("title") or first_title(body),
        "summary": question,
        "unit_id": meta.get("unit_id") or "",
        "format": "markdown",
        "read": body.strip(),
        "learn": learn_md,
        "play": play_md,
        "read_en": None,
        "learn_en": None,
        "play_en": None,
        "play_tools": [],
        "checkpoints": [],
        "body_locale": "zh",
        "original_url": original,
        "source_path": str(path),
    }


@lru_cache(maxsize=1)
def warmup() -> tuple[str, ...]:
    """Touch the registry so import-time mistakes fail fast."""
    return tuple(spec["id"] for spec in _load_topics_file())
