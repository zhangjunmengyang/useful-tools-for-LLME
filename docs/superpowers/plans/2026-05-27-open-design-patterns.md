# Open Design Patterns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the initial Open Design restyle into a reusable workbench design system with consistent navigation, page skeletons, and restrained visual primitives.

**Architecture:** Keep visual primitives in `workbench_theme.py`, keep `app_gradio.py` focused on route composition and grouped lab metadata, and migrate the highest-traffic pages to a shared control/workspace/details structure. Regression tests assert source-level adoption so future pages do not drift back to ad hoc styling.

**Tech Stack:** Python, Gradio, Plotly, unittest, Chrome headless visual verification, Open Design `design-systems/openai` and `design-systems/default`.

---

### Task 1: Add Pattern Regression Tests

**Files:**
- Modify: `tests/test_open_design_redesign.py`

- [ ] **Step 1: Write failing tests**

Add tests that require shared layout primitives (`workbench-tool-shell`, `workbench-control-panel`, `workbench-output-panel`, `workbench-detail-panel`, `metric-strip`, `status-pill`, `code-surface`, `plot-frame`), grouped lab navigation metadata, restrained Plotly colors, and page adoption in TokenLab Playground, Generation Logits Inspector, Eval Benchmark Explorer, and Agent Trace Viewer.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m unittest tests.test_open_design_redesign -v`

Expected: FAIL because the primitives and page adoption do not exist yet.

### Task 2: Add Shared Workbench Primitives

**Files:**
- Modify: `workbench_theme.py`
- Modify: `app_gradio.py`

- [ ] **Step 1: Implement token and CSS primitives**

Add neutral chart colors, layout variables, tool shell classes, status pills, metric strips, code surfaces, empty states, plot frames, and a grouped lab rail.

- [ ] **Step 2: Render grouped lab rail**

Define `LAB_GROUPS` and render a compact grouped rail above Gradio tabs so the top navigation has an explicit information architecture without replacing Gradio routing.

### Task 3: Migrate Key Pages

**Files:**
- Modify: `token_lab/playground.py`
- Modify: `generation_lab/logits_inspector.py`
- Modify: `eval_lab/benchmark_explorer.py`
- Modify: `agent_trace_lab/trace_viewer.py`

- [ ] **Step 1: Apply shared page hero and tool shell markup**

Use `workbench-page-hero`, `workbench-tool-shell`, `workbench-control-panel`, `workbench-output-panel`, and `workbench-detail-panel` classes on the visible page structure.

- [ ] **Step 2: Preserve existing interactions**

Do not change event signatures or model-loading behavior. Only add class wrappers and replace ad hoc page header markup where needed.

### Task 4: Verify

**Files:**
- Run only

- [ ] **Step 1: Unit regression**

Run: `.venv/bin/python -m unittest tests.test_open_design_redesign -v`

Expected: PASS.

- [ ] **Step 2: Syntax check**

Run: `.venv/bin/python -m compileall app_gradio.py workbench_theme.py token_lab generation_lab eval_lab agent_trace_lab tests`

Expected: PASS.

- [ ] **Step 3: App creation**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'\nimport app_gradio\nprint('import ok')\napp = app_gradio.create_app()\nprint('create_app ok')\nPY`

Expected: prints both `import ok` and `create_app ok`.

- [ ] **Step 4: Visual QA**

Run the app with `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python app_gradio.py`, capture TokenLab and Eval Lab screenshots with headless Chrome, and confirm core content renders with no hidden-page model downloads on startup.
