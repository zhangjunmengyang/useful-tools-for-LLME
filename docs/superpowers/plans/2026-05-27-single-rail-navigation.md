# Single Rail Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current rail-plus-double-tab navigation with a single left index rail and one hidden Gradio page switcher.

**Architecture:** Add page registry helpers in `app_gradio.py` so each page has one id, label, group, description, render function, and lazy-load binding. Replace nested `gr.Tabs` with one `gr.Tabs(elem_classes=["workbench-page-switcher"])` whose tab headers are visually hidden; render navigation buttons in `render_navigation_rail()` from `workbench_theme.py`.

**Tech Stack:** Python, Gradio, unittest, headless Chrome visual QA, Open Design, Impeccable-inspired anti-pattern cleanup.

---

### Task 1: Add Failing Navigation Tests

**Files:**
- Modify: `tests/test_open_design_redesign.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert `app_gradio.py` has `PAGE_REGISTRY`, `render_page_navigation`, and `workbench-page-switcher`, and does not call `render_lab_group_rail`. Add theme tests for `.workbench-app-layout`, `.workbench-navigation-rail`, `.workbench-nav-button`, and `.workbench-page-switcher > .tab-nav`.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m unittest tests.test_open_design_redesign -v`

Expected: FAIL because the app still renders grouped rail plus nested tabs.

### Task 2: Replace App Routing Shell

**Files:**
- Modify: `app_gradio.py`
- Modify: `workbench_theme.py`

- [ ] **Step 1: Add navigation rail helper**

Implement `render_page_navigation(pages)` in `workbench_theme.py` using grouped buttons with `data-page-id` attributes.

- [ ] **Step 2: Refactor page registration**

Create `PAGE_REGISTRY` in `app_gradio.py`. Render a single hidden `gr.Tabs(selected="token_playground", elem_classes=["workbench-page-switcher"])`; each registry item becomes one `gr.Tab`.

- [ ] **Step 3: Preserve lazy loading**

Bind each page `load_fn` to both the hidden page tab and first-page group tab where needed. Keep `_cached_load_handler()` unchanged.

### Task 3: Update CSS

**Files:**
- Modify: `workbench_theme.py`

- [ ] **Step 1: Remove grouped-card rail styling from the primary shell**

Keep old class names harmless for compatibility, but use `.workbench-app-layout` and `.workbench-navigation-rail` for the real shell.

- [ ] **Step 2: Hide the page switcher tab bar**

Style `.workbench-page-switcher > .tab-nav` so Gradio no longer shows a second row of tabs.

### Task 4: Verify

**Files:**
- Run only

- [ ] **Step 1: Unit regression**

Run: `.venv/bin/python -m unittest tests.test_open_design_redesign -v`

Expected: PASS.

- [ ] **Step 2: Syntax and app construction**

Run compileall and `app_gradio.create_app()`.

- [ ] **Step 3: Visual QA**

Run the app, capture the homepage and Eval Lab page with headless Chrome, and verify one left navigation rail plus one content pane, with no visible double tab rows.
