---
name: LLM Tools Workbench
version: 2026-05-29
source:
  - https://github.com/VoltAgent/awesome-design-md
  - design-md/vercel/DESIGN.md
  - design-md/linear.app/DESIGN.md
  - design-md/raycast/DESIGN.md
colors:
  canvas: "#ffffff"
  surface: "#fafafa"
  surfaceSoft: "#f5f5f5"
  ink: "#171717"
  inkStrong: "#000000"
  body: "#4d4d4d"
  mute: "#888888"
  hairline: "#ebebeb"
  hairlineSoft: "#f5f5f5"
  accent: "#171717"
  link: "#0070f3"
typography:
  body:
    fontFamily: Inter
    fontSize: 14px
    lineHeight: 20px
  compact:
    fontFamily: Inter
    fontSize: 12px
    lineHeight: 16px
  code:
    fontFamily: JetBrains Mono
radius:
  sm: 6px
  md: 8px
  lg: 10px
spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
---

# DESIGN.md

This file gives agents persistent design context for LLM Tools Workbench.

## Direction

Use `VoltAgent/awesome-design-md` as the source style library. For this product, the relevant references are Vercel, Linear, and Raycast:

- Vercel: near-white canvas, black primary actions, hairline borders, restrained developer-platform chrome.
- Linear: precise density, compact panels, low elevation, technical product confidence.
- Raycast: command-surface navigation and fast launcher behavior.

Do not use marketing heroes, decorative gradients, stacked summary cards, or large explanatory blocks.

## Navigation

The app opens with one compact command surface:

- `render_command_header`: one short title row plus compact metadata.
- `workbench-command-surface`: the only top-level navigation container.
- `Task Area`: compact dropdown filter, not a segmented tab group.
- `Tool`: searchable dropdown for the selected task area.
- `workbench-page-router`: hidden Gradio tabs used only as the routing mechanism.

Avoid:

- Long scrolling sidebars.
- Horizontal scrolling tab strips for top-level routing.
- Double-row tab systems.
- Duplicated group summary cards above the real selector.
- Any layout that forces users to scan every tool before working.

## Layout

Keep the app dense and work-focused:

- Header height should stay compact.
- The command surface should sit close to the header and content.
- Page hero strips should be short, not card-like.
- Tool pages use `workbench-tool-shell`.
- Inputs live in `workbench-control-panel`.
- Results live in `workbench-output-panel`.

Cards are allowed only for real panels, repeated items, or framed tools. Do not put navigation cards above navigation controls.

## Tokens

Use black, white, neutral gray, and one blue link/state accent:

- Primary ink and selected state: `#171717`.
- Link and informational state: `#0070f3`.
- Hairline border: `#ebebeb`.
- Soft surface: `#fafafa` or `#f5f5f5`.

Do not reintroduce HuggingFace orange, purple gradients, green OpenAI-style accent, beige themes, or one-off saturated palettes.

## Typography

Use Inter for interface text and JetBrains Mono for code, IDs, JSON, and numeric-heavy surfaces. Headings are compact and functional. No negative letter spacing.

## Responsive

Desktop: command surface remains a compact full-width bar with two fields: task area and tool.

Mobile: command surface stacks into two dropdowns. The selected page follows immediately after the selector.

## Agent Rules

1. User-visible UI text must be English.
2. No emoji in project UI.
3. Preserve the command-surface pattern unless the whole information architecture is intentionally replaced.
4. Prefer fewer containers and tighter rhythm over stacked explanatory cards.
5. Verify layout changes with browser screenshots on desktop and mobile widths.
