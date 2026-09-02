"""Copy Omni / WM / CL course bodies into content/<topic>/.

Source of truth is the mem-learn tree at /Users/zhangjunmengyang/project/learn-omni.
Does not copy training artifacts, node_modules, .next, or _legacy binaries.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

MEM_LEARN = Path("/Users/zhangjunmengyang/project/learn-omni")
REPO_ROOT = Path(__file__).resolve().parents[1]

UNIT_RE = re.compile(
    r'id:\s*"(?P<id>[\w-]+)"\s*,\s*order:\s*(?P<order>\d+)\s*,\s*title:\s*"(?P<title>[^"]+)"\s*,\s*question:\s*"(?P<question>[^"]*)"',
)
LESSON_HEAD_RE = re.compile(
    r'\{\s*id:\s*"(?P<id>\d+)"\s*,\s*slug:\s*"(?P<slug>[^"]+)"\s*,\s*shortTitle:\s*"(?P<title>[^"]+)"\s*,\s*unit:\s*unitById(?:\.(?P<unit_dot>[\w-]+)|\[\s*"(?P<unit_br>[\w-]+)"\s*\])',
    re.S,
)
QUESTION_RE = re.compile(r'essentialQuestion:\s*"((?:\\.|[^"\\])*)"')
OUTCOMES_RE = re.compile(r"outcomes:\s*\[(.*?)\]", re.S)
STRING_RE = re.compile(r'"((?:\\.|[^"\\])*)"')

COURSES = (
    {
        "id": "omni",
        "title": "Omni",
        "title_en": "Omni",
        "summary": "从 MiniMind-O 到现代全模态系统：文本、语音、图像和动作怎么接在同一条链路上。",
        "summary_en": "From MiniMind-O to modern omni systems: how text, speech, images, and actions share one stack.",
        "source": MEM_LEARN / "learn-omni",
        "expected": 60,
        "copy_experiments": True,
        "extra_files": (
            "web/lib/glossary.ts",
            "web/lib/practice-data.ts",
            "web/lib/course-data.ts",
            "web/content/course-manifest.json",
        ),
        "extra_dirs": (
            "web/components/labs",
            "web/lib/lesson-diagrams",
            "web/lib/expansion-meta",
        ),
    },
    {
        "id": "wm",
        "title": "世界模型",
        "title_en": "World Models",
        "summary": "从 2018 年 World Models 复现，走到潜空间、生成式、JEPA 和具身桌宠。",
        "summary_en": "From the 2018 World Models reproduction through latent spaces, generative models, JEPA, and an embodied desktop pet.",
        "source": MEM_LEARN / "learn-wm",
        "expected": 45,
        "copy_experiments": False,
        "extra_files": (
            "web/lib/glossary.ts",
            "web/lib/practice-data.ts",
            "web/lib/course-data.ts",
            "web/lib/embodiment-score.ts",
            "web/content/course-manifest.json",
        ),
        "extra_dirs": (
            "web/components/labs",
            "web/lib/lesson-diagrams",
        ),
    },
    {
        "id": "cl",
        "title": "持续学习",
        "title_en": "Continual Learning",
        "summary": "灾难性遗忘怎么量、四类补丁怎么补，以及大模型接龙和在岗学习。",
        "summary_en": "How to measure catastrophic forgetting, four families of fixes, and what changes when the model is already large.",
        "source": MEM_LEARN / "learn-cl",
        "expected": 24,
        "copy_experiments": True,
        "extra_files": (
            "web/lib/glossary.ts",
            "web/lib/practice-data.ts",
            "web/lib/course-data.ts",
            "web/lib/extra-experiments.ts",
            "web/lib/gpu-recipes.ts",
            "web/content/course-manifest.json",
        ),
        "extra_dirs": (
            "web/components/labs",
            "web/lib/lesson-diagrams",
        ),
    },
)

SKIP_DIR_NAMES = {
    "artifacts",
    "node_modules",
    ".next",
    ".git",
    "__pycache__",
    ".venv",
    "dist",
}


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def read_course_data(meta_path: Path) -> tuple[list[dict], dict[str, dict]]:
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
        }
        for match in UNIT_RE.finditer(units_chunk)
    ]
    lessons: dict[str, dict] = {}
    for match in LESSON_HEAD_RE.finditer(text):
        block_start = match.start()
        block_end = text.find("\n  },", block_start)
        block = text[block_start : block_end if block_end != -1 else block_start + 2500]
        question_match = QUESTION_RE.search(block)
        outcomes_match = OUTCOMES_RE.search(block)
        outcomes = []
        if outcomes_match:
            outcomes = [item.replace('\\"', '"') for item in STRING_RE.findall(outcomes_match.group(1))]
        lessons[match.group("id")] = {
            "number": match.group("id"),
            "slug": match.group("slug"),
            "title": match.group("title"),
            "unit_id": match.group("unit_dot") or match.group("unit_br"),
            "question": question_match.group(1).replace('\\"', '"') if question_match else "",
            "outcomes": outcomes,
        }
    return units, lessons


def copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = {name for name in names if name in SKIP_DIR_NAMES}
        skipped.update(name for name in names if name.endswith(".pyc"))
        return skipped

    shutil.copytree(src, dest, ignore=ignore)


def copy_experiments(src_root: Path, dest_root: Path) -> None:
    experiments = src_root / "experiments"
    if not experiments.is_dir():
        return
    dest = dest_root / "experiments"
    copy_tree(experiments, dest)
    artifacts = dest / "artifacts"
    if artifacts.exists():
        shutil.rmtree(artifacts)


def write_lesson(path: Path, meta: dict, dest: Path) -> None:
    body = path.read_text(encoding="utf-8")
    if body.startswith("---\n"):
        raise RuntimeError(f"unexpected frontmatter already present: {path}")
    lesson_id = path.stem
    title = meta.get("title") or lesson_id
    summary = meta.get("question") or ""
    unit = meta.get("unit_id") or "other"
    lines = [
        "---",
        f"id: {lesson_id}",
        f"title: {yaml_quote(title)}",
        f"summary: {yaml_quote(summary)}",
        f"unit: {unit}",
        "play_tools: []",
    ]
    outcomes = meta.get("outcomes") or []
    if outcomes:
        lines.append("checkpoints:")
        lines.extend(f"  - {yaml_quote(item)}" for item in outcomes)
    else:
        lines.append("checkpoints: []")
    lines.extend(["---", "", body.lstrip("\n")])
    dest.write_text("\n".join(lines), encoding="utf-8")


def vendor_course(spec: dict) -> Path:
    source: Path = spec["source"]
    dest = REPO_ROOT / "content" / spec["id"]
    lessons_src = source / "web" / "content" / "lessons"
    if dest.exists():
        shutil.rmtree(dest)
    lessons_dest = dest / "lessons"
    lessons_dest.mkdir(parents=True)

    units, by_number = read_course_data(source / "web" / "lib" / "course-data.ts")
    paths = sorted(lessons_src.glob("*.md"))
    if len(paths) != spec["expected"]:
        raise RuntimeError(f"{spec['id']}: expected {spec['expected']} lessons, found {len(paths)}")
    if len(by_number) != spec["expected"]:
        raise RuntimeError(f"{spec['id']}: expected {spec['expected']} meta rows, found {len(by_number)}")

    for path in paths:
        number = re.match(r"^(\d+)", path.stem)
        key = number.group(1) if number else path.stem
        write_lesson(path, by_number.get(key, {}), lessons_dest / path.name)

    course = {
        "id": spec["id"],
        "title": spec["title"],
        "title_en": spec["title_en"],
        "summary": spec["summary"],
        "summary_en": spec["summary_en"],
        "units": [
            {
                "id": unit["id"],
                "title": unit["title"],
                "question": unit["question"],
            }
            for unit in units
        ],
    }
    (dest / "course.json").write_text(
        json.dumps(course, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for relative in spec["extra_files"]:
        src = source / relative
        if src.is_file():
            target = dest / Path(relative).name
            shutil.copy2(src, target)

    for relative in spec["extra_dirs"]:
        src = source / relative
        if src.is_dir():
            copy_tree(src, dest / Path(relative).name)

    if spec["copy_experiments"]:
        copy_experiments(source, dest)

    copied = sorted(p.name for p in lessons_dest.glob("*.md"))
    print(f"{spec['id']}: {len(copied)} lessons -> {dest}")
    return dest


def main() -> None:
    if not MEM_LEARN.is_dir():
        raise SystemExit(f"missing mem-learn tree: {MEM_LEARN}")
    for spec in COURSES:
        vendor_course(spec)


if __name__ == "__main__":
    main()
