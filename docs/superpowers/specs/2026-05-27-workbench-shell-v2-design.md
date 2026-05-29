# Workbench Shell v2 Design

## Goal

Refine the workbench shell after the single-rail redesign so it reads as a mature engineering tool: calmer page headers, grouped navigation, better mobile behavior, and clearer interactive states.

## Design Sources

- Open Design / OpenAI: near-monochrome palette, generous whitespace, hairline borders, restrained typography.
- Impeccable: flat surfaces at rest, every element earns its place, avoid nested card chrome and generic AI dashboard tropes.
- WCAG 2.2: preserve keyboard access, visible focus, and practical touch targets.
- Fluent / Apple / Material navigation patterns: keep navigation scan-friendly, clear, and responsive instead of relying on stacked tabs.

## Scope

1. Keep Gradio native Tabs as the switching mechanism for reliability.
2. Add generated CSS group labels to the page switcher based on the filtered page registry.
3. Convert page hero cards into compact header strips.
4. Treat inner tabs as lightweight segmented controls inside the active work surface.
5. On mobile, turn the long left rail into a horizontal picker above the content and collapse descriptive group summaries.

## Non-Goals

- No new JavaScript router.
- No new component framework.
- No decorative gradients, orbs, glassmorphism, or marketing hero treatment.
- No rewrite of each individual lab page beyond shared shell styling.

## Success Criteria

- Desktop shows one navigation system with group labels and no top-level double tab row.
- Mobile does not force users through a full-height vertical page list before content.
- Page headers no longer read as framed cards.
- Inner tabs remain visible but lighter than primary navigation.
- Unit tests and screenshot QA verify shell behavior.
