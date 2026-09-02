"""Catalog of Gradio lab pages for the learn shell."""

from __future__ import annotations

from typing import Any

from app_gradio import LAB_NAV_LABELS, get_available_pages


def list_lab_pages(*, labs_mounted: bool) -> dict[str, Any]:
    """Return every reachable Gradio page, including embed URLs."""
    pages = []
    for page in get_available_pages():
        page_id = page["id"]
        pages.append(
            {
                "id": page_id,
                "label": page["label"],
                "lab": page["lab"],
                "lab_label": LAB_NAV_LABELS.get(page["lab"], page["lab"]),
                "group": page["group"],
                "group_description": page["group_description"],
                "module": page["module"],
                "embed_url": f"/labs/?lab={page_id}",
            }
        )
    return {
        "mounted": labs_mounted,
        "embed_root": "/labs/",
        "pages": pages,
    }
