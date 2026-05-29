# Page Polish Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring representative legacy lab pages into the same Open Design workbench layout used by the shell.

**Architecture:** Keep page logic and event bindings unchanged. Only rearrange Gradio components into shared page hero, tool shell, control panel, output panel, detail panel, plot frame, and metric strip primitives.

**Tech Stack:** Python, Gradio, unittest, compileall, headless Chrome visual QA.

---

### Task 1: Add Regression Coverage

**Files:**
- Modify: `tests/test_open_design_redesign.py`

- [ ] Add assertions that representative legacy pages use `workbench-page-hero`, `workbench-tool-shell`, `workbench-control-panel`, and `workbench-output-panel`.
- [ ] Add assertions that these pages do not use top-level `gr.Markdown("# ...")` headings.
- [ ] Run `.venv/bin/python -m unittest tests.test_open_design_redesign -v`.
- [ ] Confirm the new assertions fail before implementation.

### Task 2: Polish Representative Pages

**Files:**
- Modify: `embedding_lab/model_comparison.py`
- Modify: `data_lab/hf_dataset_viewer.py`
- Modify: `model_lab/memory_estimator.py`
- Modify: `finetune_lab/training_cost_estimator.py`

- [ ] Add compact `.workbench-page-hero` headers.
- [ ] Wrap controls in `.workbench-control-panel`.
- [ ] Wrap outputs in `.workbench-output-panel`.
- [ ] Place charts/tables in `.plot-frame` or `.workbench-detail-panel`.
- [ ] Preserve existing component variables, inputs, outputs, event handlers, and load returns.

### Task 3: Verify

**Commands:**
- `.venv/bin/python -m unittest tests.test_open_design_redesign -v`
- `.venv/bin/python -m py_compile embedding_lab/model_comparison.py data_lab/hf_dataset_viewer.py model_lab/memory_estimator.py finetune_lab/training_cost_estimator.py`
- `.venv/bin/python -m compileall app_gradio.py workbench_theme.py tests/test_open_design_redesign.py embedding_lab/model_comparison.py data_lab/hf_dataset_viewer.py model_lab/memory_estimator.py finetune_lab/training_cost_estimator.py`
- Headless Chrome screenshot on at least two changed pages.

**Expected:** Tests pass, files compile, app still switches pages, and the changed pages visually match the workbench shell.
