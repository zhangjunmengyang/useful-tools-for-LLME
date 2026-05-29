# Workbench Shell v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Gradio shell from a merely single-rail layout into a polished engineering workbench shell with grouped navigation, calmer page headers, lighter inner tabs, and mobile-friendly navigation.

**Architecture:** Keep Gradio native Tabs as the routing primitive. Generate CSS for group labels from `PAGE_REGISTRY` after optional pages are filtered, then refine shared CSS in `workbench_theme.py` so all pages inherit the shell improvements.

**Tech Stack:** Python, Gradio, CSS, unittest, headless Chrome CDP screenshots.

---

### Task 1: Lock Shell v2 Behavior With Tests

**Files:**
- Modify: `tests/test_open_design_redesign.py`

- [ ] Add assertions for `render_page_switcher_group_styles`, generated `nth-of-type` group labels, compact page headers, mobile horizontal picker, focus-visible states, and lightweight inner tab styling.
- [ ] Run `.venv/bin/python -m unittest tests.test_open_design_redesign -v`.
- [ ] Confirm the new assertions fail before implementation.

### Task 2: Generate Grouped Navigation CSS

**Files:**
- Modify: `workbench_theme.py`
- Modify: `app_gradio.py`

- [ ] Add `render_page_switcher_group_styles(pages)` to compute group start positions after optional pages are filtered.
- [ ] Inject the generated `<style>` tag immediately before the page switcher.
- [ ] Keep the native Gradio tab buttons as the only switching control.

### Task 3: Refine Shared Shell CSS

**Files:**
- Modify: `workbench_theme.py`

- [ ] Raise navigation target height to 44px.
- [ ] Add `:focus-visible` treatment for primary rail and inner segmented controls.
- [ ] Convert `.workbench-page-hero` and `.main-header` from cards into compact page header strips.
- [ ] Style inner page tabs as lighter segmented controls under the page header.
- [ ] On mobile, collapse descriptive group summaries and make the page switcher horizontal instead of a tall vertical rail.

### Task 4: Verify

**Commands:**
- `.venv/bin/python -m unittest tests.test_open_design_redesign -v`
- `.venv/bin/python -m compileall app_gradio.py workbench_theme.py tests/test_open_design_redesign.py`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY' ... create_app ... PY`
- Headless Chrome desktop screenshot for home and Eval page.
- Headless Chrome mobile screenshot for the nav picker.

**Expected:** Tests pass, app builds, screenshots show one primary navigation, grouped rail labels, compact page headers, and no mobile full-height vertical nav list before content.
