# Open Design Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Gradio app shell and shared visual system around `nexu-io/open-design` while keeping the existing lab modules functional.

**Architecture:** Extract shared visual responsibility into `workbench_theme.py`, keep `app_gradio.py` focused on routing, and use small targeted page patches for the most visible rule violations. Tests assert source-backed theme use and guard against broken optional lab imports.

**Tech Stack:** Python, Gradio, Plotly, unittest, Open Design `design-systems/openai` and `design-systems/default`.

---

### Task 1: Add Open Design Regression Tests

**Files:**
- Create: `tests/test_open_design_redesign.py`

- [x] **Step 1: Write failing tests**

Create tests that assert `app_gradio.py` imports shared theme helpers, `workbench_theme.py` owns Open Design token references, and `inference_lab` is not imported unconditionally.

- [x] **Step 2: Run tests to verify failure**

Run: `python -m unittest tests.test_open_design_redesign -v`

Expected: FAIL because the old app still owns HuggingFace CSS and imports `inference_lab` directly.

### Task 2: Extract Shared Theme

**Files:**
- Create: `workbench_theme.py`
- Modify: `app_gradio.py`

- [ ] **Step 1: Add Open Design theme module**

Define `CUSTOM_CSS`, `CUSTOM_THEME`, `TOKEN_COLORS`, `render_app_header()`, and `configure_plotly_theme()` in `workbench_theme.py` using `open-design/design-systems/openai` tokens.

- [ ] **Step 2: Wire theme into Blocks**

Import the shared helpers in `app_gradio.py`, call `configure_plotly_theme()`, render the class-based header, and pass `theme=CUSTOM_THEME` plus `css=CUSTOM_CSS` to `gr.Blocks`.

- [ ] **Step 3: Remove launch theme arguments**

Keep `app.launch()` focused on server settings only.

### Task 3: Guard Optional Labs

**Files:**
- Modify: `app_gradio.py`

- [ ] **Step 1: Add optional lab detection**

Add `optional_lab_available(package_name: str) -> bool` using `importlib.util.find_spec`.

- [ ] **Step 2: Guard InferenceLab route**

Only render InferenceLab when `optional_lab_available("inference_lab")` returns true.

### Task 4: Clean Visible Eval Pages

**Files:**
- Modify: `eval_lab/benchmark_explorer.py`
- Modify: `eval_lab/llm_judge.py`
- Modify: `eval_lab/eval_pipeline.py`

- [ ] **Step 1: Replace page headers**

Replace emoji-based `main-header` snippets with `workbench-page-hero` markup.

- [ ] **Step 2: English UI text**

Translate tab titles, visible labels, button text, chart titles, and empty-state text that appear in the primary Eval Lab workflows.

### Task 5: Verify

**Files:**
- Run only

- [ ] **Step 1: Unit regression**

Run: `python -m unittest tests.test_open_design_redesign -v`

Expected: PASS.

- [ ] **Step 2: Syntax check**

Run: `python -m compileall app_gradio.py workbench_theme.py eval_lab`

Expected: PASS.

- [ ] **Step 3: Runtime check when dependencies are available**

Run: `python app_gradio.py`

Expected: Gradio starts on `http://localhost:7860`; if dependencies are missing, report the missing package instead of claiming runtime validation.
