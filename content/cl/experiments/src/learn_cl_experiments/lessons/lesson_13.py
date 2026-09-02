from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


def _tokens(text: str) -> set[str]:
    return {piece for piece in text.lower().replace(",", " ").split() if piece}


class OverwriteMemory:
    """Mem0-style: same (subject, attribute) key keeps only the latest value."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def write(self, subject: str, attribute: str, value: str) -> None:
        self.store[(subject, attribute)] = value

    def read(self, subject: str, attribute: str) -> str | None:
        return self.store.get((subject, attribute))


class AppendMemory:
    """Diary-style: conflict facts coexist."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def write(self, subject: str, attribute: str, value: str) -> None:
        self.rows.append((subject, attribute, value))

    def read_all(self, subject: str, attribute: str) -> list[str]:
        return [value for sub, attr, value in self.rows if sub == subject and attr == attribute]


class GraphMemory:
    """HippoRAG-style: overwrite the same (head, relation), then hop."""

    def __init__(self) -> None:
        self.edge: dict[tuple[str, str], str] = {}

    def write(self, head: str, relation: str, tail: str) -> None:
        self.edge[(head, relation)] = tail

    def hop(self, start: str, relations: list[str]) -> str | None:
        node: str | None = start
        for relation in relations:
            if node is None:
                return None
            node = self.edge.get((node, relation))
        return node


class FlatRAG:
    def __init__(self) -> None:
        self.docs: list[str] = []

    def add(self, text: str) -> None:
        self.docs.append(text)

    def retrieve(self, query: str) -> str | None:
        query_tokens = _tokens(query)
        if not query_tokens or not self.docs:
            return None
        scored = []
        for doc in self.docs:
            overlap = len(query_tokens & _tokens(doc))
            scored.append((overlap, doc))
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_overlap, best_doc = scored[0]
        if best_overlap <= 0:
            return None
        ties = [doc for overlap, doc in scored if overlap == best_overlap]
        if len(ties) > 1:
            return None
        return best_doc


def run() -> dict[str, Any]:
    overwrite = OverwriteMemory()
    append = AppendMemory()
    graph = GraphMemory()
    rag = FlatRAG()
    working: list[str] = []
    semantic = {"xiaowang": "engineer", "xiaoli": "designer"}

    overwrite.write("xiaowang", "seat", "A3")
    append.write("xiaowang", "seat", "A3")
    graph.write("xiaowang", "project", "beiji")
    graph.write("beiji", "floor", "2F")
    graph.write("xiaowang", "seat", "A3")
    rag.add("xiaowang joins project beiji")
    rag.add("project beiji sits on floor 2F")
    working.append("today we moved desks")

    overwrite.write("xiaowang", "seat", "B7")
    append.write("xiaowang", "seat", "B7")
    graph.write("xiaowang", "seat", "B7")

    overwrite_seat = overwrite.read("xiaowang", "seat")
    append_seats = append.read_all("xiaowang", "seat")
    graph_seat = graph.hop("xiaowang", ["seat"])
    graph_floor = graph.hop("xiaowang", ["project", "floor"])
    flat_floor_doc = rag.retrieve("xiaowang floor")
    working.clear()

    checks = {
        "overwrite_returns_new_seat": overwrite_seat == "B7",
        "overwrite_drops_old_seat": overwrite.read("xiaowang", "seat") != "A3",
        "append_keeps_both_seats": append_seats == ["A3", "B7"],
        "graph_overwrites_same_relation": graph_seat == "B7",
        "graph_multihop_reaches_floor": graph_floor == "2F",
        "flat_bow_cannot_compose_hops": flat_floor_doc is None,
        "semantic_directory_survives_working_flush": (
            not working and semantic["xiaowang"] == "engineer"
        ),
    }
    return {
        "summary": (
            "覆盖规则把小王座位从 A3 写成 B7；追加日记仍同时保留两个值；"
            "知识图沿 project→floor 跳到 2F，平面词袋检索在两跳上打平所以答不出。"
            "工作记忆清空后语义名录仍在。覆盖失败阈值：读回不是 B7。"
        ),
        "metrics": {
            "overwrite_seat": overwrite_seat or "",
            "append_seat_count": len(append_seats),
            "append_unique_seats": len(set(append_seats)),
            "graph_seat": graph_seat or "",
            "graph_floor": graph_floor or "",
            "flat_retrieved": flat_floor_doc or "",
            "working_after_flush": len(working),
            "semantic_size": len(semantic),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="13",
    title="把日记写在模型外面",
    question="冲突事实写入后，覆盖、并存、图跳转各自召回什么？",
    run=run,
)
