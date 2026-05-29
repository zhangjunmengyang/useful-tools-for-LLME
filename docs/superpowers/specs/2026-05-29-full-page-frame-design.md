# Full Page Frame Design

## Goal

Give every registered workbench page a consistent Open Design shell, including pages that have not yet been manually refactored.

## Approach

Pages that already include the shared workbench primitives keep their native page layout. Pages that still use the older Gradio layout are wrapped by the app entrypoint in a lightweight fallback frame:

- compact page header generated from the registry
- narrow context column using `.workbench-control-panel`
- main content column using `.workbench-output-panel`

This avoids risky edits across every old page while making the full application coherent today.

## Rules

- Keep all page event wiring inside each page module unchanged.
- Do not wrap pages that already declare `workbench-page-hero`, `workbench-tool-shell`, `workbench-control-panel`, and `workbench-output-panel`.
- Keep generated user-facing text English.
- Keep fallback framing visually quiet and clearly subordinate to the page content.
- Continue manual page-by-page migration later for pages that need deeper information architecture work.
