# Full Page Frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every registered page has a coherent workbench frame without manually rewriting all legacy pages in one risky pass.

**Architecture:** Detect whether a page source already uses the shared workbench primitives. If not, wrap the page render call in a generated fallback shell from `app_gradio.py` and `workbench_theme.py`.

**Tech Stack:** Python, Gradio, unittest, CSS, headless Chrome CDP screenshots.

---

### Task 1: Write Failing Tests

**Files:**
- Modify: `tests/test_open_design_redesign.py`

- [ ] Assert that `workbench_theme.py` exposes fallback rendering helpers.
- [ ] Assert that `app_gradio.py` detects existing workbench pages and wraps legacy pages.
- [ ] Assert that known legacy pages such as `token_lab.arena` and `generation_lab.beam_visualizer` are not manually marked as migrated.

### Task 2: Implement Fallback Shell

**Files:**
- Modify: `workbench_theme.py`
- Modify: `app_gradio.py`

- [ ] Add `render_legacy_page_header(page, page_name)` and `render_legacy_page_context(page, page_name)`.
- [ ] Add `_page_uses_workbench_layout(page)`.
- [ ] Add `_render_page_with_frame(page, page_name)` and call it from `create_app`.
- [ ] Add CSS for `.workbench-legacy-page-shell`, `.workbench-legacy-page-context`, and `.workbench-legacy-output-panel`.

### Task 3: Verify

**Commands:**
- `.venv/bin/python -m unittest tests.test_open_design_redesign -v`
- `.venv/bin/python -m compileall app_gradio.py workbench_theme.py tests/test_open_design_redesign.py`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY' ... create_app ... PY`
- Headless Chrome screenshots for one legacy Token page and one legacy Generation page.

**Expected:** Tests pass; migrated pages are not double-wrapped; legacy pages render inside the generated shell.
