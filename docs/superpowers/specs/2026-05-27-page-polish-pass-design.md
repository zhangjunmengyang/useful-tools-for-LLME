# Page Polish Pass Design

## Goal

Extend the Open Design workbench language from the shell into representative pages across the older labs.

## Approach

This pass updates one high-traffic page per area instead of mechanically touching every page. Each page should use the same structure: compact page header, left control panel, right output panel, framed plot/table surfaces, and restrained detail panels.

## Pages

- EmbeddingLab / Model Comparison
- DataLab / Dataset Viewer
- ModelLab / Memory Estimator
- FineTuneLab / Training Cost

## Design Rules

- Use `.workbench-page-hero` for page context.
- Use `.workbench-tool-shell`, `.workbench-control-panel`, and `.workbench-output-panel` for main task layout.
- Put charts and data tables inside `.plot-frame` or `.workbench-detail-panel`.
- Keep user-visible text in English.
- Avoid decorative cards and large Markdown-only headings.
- Preserve existing event wiring and load behavior.
