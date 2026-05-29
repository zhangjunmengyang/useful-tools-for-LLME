"""Workbench 共享视觉系统。"""

from html import escape

import gradio as gr
import plotly.io as pio


# Source-backed tokens:
# - https://github.com/VoltAgent/awesome-design-md
# - design-md/vercel/DESIGN.md
# - design-md/linear.app/DESIGN.md
# - design-md/raycast/DESIGN.md
OPEN_DESIGN_COLORS = {
    "bg": "#ffffff",
    "surface": "#fafafa",
    "surface_warm": "#f5f5f5",
    "fg": "#171717",
    "fg_2": "#000000",
    "muted": "#4d4d4d",
    "meta": "#888888",
    "stone": "#a1a1a1",
    "border": "#ebebeb",
    "border_soft": "#f5f5f5",
    "accent": "#171717",
    "accent_hover": "#000000",
    "success": "#0070f3",
    "warning": "#f5a623",
    "danger": "#ee0000",
    "info": "#0070f3",
}

TOKEN_COLORS = [
    "#e8f5f0",
    "#eef2ff",
    "#f5f5f5",
    "#fff7ed",
    "#fef2f2",
    "#eff6ff",
    "#f0fdf4",
    "#faf5ff",
    "#fefce8",
    "#f8fafc",
    "#fdf2f8",
    "#ecfeff",
]

PLOTLY_COLORWAY = [
    OPEN_DESIGN_COLORS["accent"],
    OPEN_DESIGN_COLORS["info"],
    OPEN_DESIGN_COLORS["fg_2"],
    OPEN_DESIGN_COLORS["muted"],
    OPEN_DESIGN_COLORS["meta"],
    OPEN_DESIGN_COLORS["stone"],
]

LAB_GROUPS = [
    {
        "label": "Core Mechanics",
        "description": "Inspect tokenization, embeddings, generation, and model internals.",
        "labs": ["TokenLab", "EmbeddingLab", "GenerationLab", "InterpretabilityLab"],
    },
    {
        "label": "Knowledge & Data",
        "description": "Prepare datasets and reason about retrieval behavior.",
        "labs": ["DataLab", "RAGLab"],
    },
    {
        "label": "Model Ops",
        "description": "Estimate memory, fine-tuning cost, and inference constraints.",
        "labs": ["ModelLab", "FineTuneLab", "InferenceLab"],
    },
    {
        "label": "Evaluation",
        "description": "Review agent traces and compare evaluation pipelines.",
        "labs": ["Agent Trace Lab", "Eval Lab"],
    },
]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --od-bg: #ffffff;
    --od-surface: #fafafa;
    --od-surface-warm: #f5f5f5;
    --od-fg: #171717;
    --od-fg-2: #000000;
    --od-muted: #4d4d4d;
    --od-meta: #888888;
    --od-border: #ebebeb;
    --od-border-soft: #f5f5f5;
    --od-accent: #171717;
    --od-accent-hover: #000000;
    --od-success: #0070f3;
    --od-warning: #f5a623;
    --od-danger: #ee0000;
    --od-info: #0070f3;
    --od-stone: #a1a1a1;
    --od-radius-sm: 6px;
    --od-radius-md: 8px;
    --od-radius-lg: 10px;
    --od-radius-pill: 9999px;
    --od-nav-width: 244px;
    --od-font-body: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    --od-font-mono: 'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace;
    --od-focus-ring: 0 0 0 3px rgba(0, 112, 243, 0.14);
    --od-ease: cubic-bezier(0.16, 1, 0.3, 1);
}

* {
    letter-spacing: 0 !important;
}

body,
.gradio-container {
    background: var(--od-bg) !important;
    color: var(--od-fg) !important;
    font-family: var(--od-font-body) !important;
    font-weight: 400 !important;
}

.gradio-container {
    max-width: none !important;
    padding: 0 !important;
}

.contain {
    max-width: none !important;
}

footer {
    display: none !important;
}

/* App shell */
.workbench-shell {
    min-height: 100vh;
    background: var(--od-bg);
}

.workbench-header {
    border-bottom: 1px solid var(--od-border);
    background: rgba(255, 255, 255, 0.96);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}

.workbench-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
}

.workbench-mark {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    background: var(--od-fg);
    flex: 0 0 auto;
}

.workbench-title {
    margin: 0;
    color: var(--od-fg);
    font-size: 15px;
    font-weight: 600;
    line-height: 1.2;
}

.workbench-subtitle {
    margin: 3px 0 0;
    color: var(--od-muted);
    font-size: 12px;
    line-height: 1.35;
}

.workbench-header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.workbench-pill,
.chip,
.pill {
    display: inline-flex;
    align-items: center;
    border-radius: var(--od-radius-pill);
    background: var(--od-surface);
    border: 1px solid var(--od-border);
    color: var(--od-muted);
    padding: 4px 8px;
    font-size: 12px;
    font-weight: 500;
    line-height: 1;
}

.workbench-page {
    padding: 24px 32px 36px;
}

.workbench-app-layout {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 14px 24px 32px !important;
}

.workbench-main-surface {
    min-width: 0 !important;
}

.workbench-page-switcher {
    display: block !important;
    min-width: 0 !important;
    width: 100% !important;
}

.workbench-command-surface {
    background: var(--od-bg) !important;
    border: 1px solid var(--od-border) !important;
    border-radius: var(--od-radius-md) !important;
    display: grid !important;
    gap: 10px !important;
    margin-bottom: 16px !important;
    padding: 12px !important;
}

.workbench-command-header {
    align-items: center;
    border-bottom: 1px solid var(--od-border);
    display: grid;
    gap: 12px;
    grid-template-columns: minmax(0, 1fr) auto;
    padding-bottom: 10px;
}

.workbench-command-title {
    color: var(--od-fg);
    font-size: 14px;
    font-weight: 600;
    line-height: 1.25;
    margin: 0;
}

.workbench-command-copy {
    color: var(--od-muted);
    font-size: 12px;
    line-height: 1.45;
    margin: 3px 0 0;
}

.workbench-command-meta {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    justify-content: flex-end;
}

.workbench-launcher-controls {
    align-items: end !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    display: grid !important;
    gap: 10px !important;
    grid-template-columns: 1fr !important;
    padding: 0 !important;
}

.workbench-command-surface .workbench-launcher-controls > .form {
    align-items: end !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    display: grid !important;
    gap: 10px !important;
    grid-template-columns: minmax(180px, 0.42fr) minmax(320px, 1fr) !important;
    padding: 0 !important;
    width: 100% !important;
}

.workbench-group-selector,
.workbench-tool-selector {
    max-width: none !important;
    min-width: 0 !important;
    width: 100% !important;
}

.workbench-tool-selector {
    max-width: none !important;
    width: 100% !important;
}

.workbench-page-router > .tab-wrapper {
    display: none !important;
}

.workbench-page-router > .tabitem {
    min-width: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}

.workbench-layout {
    max-width: 1200px;
    margin: 0 auto;
}

.workbench-page-hero,
.main-header {
    background: transparent !important;
    border: 0 !important;
    border-bottom: 1px solid var(--od-border) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    margin-bottom: 14px !important;
    padding: 0 0 12px !important;
}

.workbench-page-hero h1,
.main-header h1,
.workbench-page-hero h2,
.main-header h2 {
    margin: 0 !important;
    color: var(--od-fg) !important;
    font-size: 22px !important;
    line-height: 1.2 !important;
    font-weight: 600 !important;
}

.workbench-page-hero p,
.main-header p {
    margin: 5px 0 0 !important;
    color: var(--od-muted) !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    max-width: 76ch !important;
}

.workbench-tool-shell {
    align-items: stretch !important;
    display: grid !important;
    gap: 12px !important;
    grid-template-columns: minmax(240px, var(--od-nav-width)) minmax(0, 1fr) !important;
    margin-top: 10px !important;
}

.workbench-page-switcher > .tabitem .tab-wrapper {
    border: 0 !important;
    display: block !important;
    height: auto !important;
    margin-bottom: 16px !important;
    max-height: none !important;
    overflow: visible !important;
    padding: 0 !important;
    position: static !important;
    width: 100% !important;
}

.workbench-page-switcher > .tabitem .tab-wrapper .tab-container[role="tablist"] {
    align-items: center !important;
    border-bottom: 1px solid var(--od-border) !important;
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    height: auto !important;
    overflow-x: auto !important;
    padding: 0 0 6px !important;
    width: 100% !important;
}

.workbench-page-switcher > .tabitem .tab-wrapper button[aria-selected] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: var(--od-muted) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    min-height: 36px !important;
    padding: 7px 12px 8px !important;
    white-space: nowrap !important;
}

.workbench-page-switcher > .tabitem .tab-wrapper button[aria-selected]:hover {
    background: var(--od-surface-warm) !important;
    color: var(--od-fg) !important;
}

.workbench-page-switcher > .tabitem .tab-wrapper button[aria-selected]:focus-visible {
    outline: 2px solid var(--od-accent) !important;
    outline-offset: 2px !important;
}

.workbench-page-switcher > .tabitem .tab-wrapper button[aria-selected].selected {
    background: transparent !important;
    border-bottom-color: var(--od-accent) !important;
    color: var(--od-accent-hover) !important;
}

.workbench-control-panel,
.workbench-output-panel,
.workbench-detail-panel,
.plot-frame {
    background: var(--od-bg) !important;
    border: 1px solid var(--od-border-soft) !important;
    border-radius: var(--od-radius-lg) !important;
    box-shadow: none !important;
    min-width: 0 !important;
    padding: 14px !important;
}

.workbench-control-panel {
    align-self: start !important;
    position: sticky !important;
    top: 16px !important;
}

.workbench-output-panel {
    min-height: 360px !important;
}

.workbench-detail-panel {
    background: var(--od-surface-warm) !important;
}

.workbench-panel-title {
    color: var(--od-fg);
    font-size: 13px;
    font-weight: 600;
    line-height: 1.35;
    margin: 0 0 10px;
}

.workbench-panel-copy {
    color: var(--od-muted);
    font-size: 13px;
    line-height: 1.55;
    margin: 0 0 14px;
}

.metric-strip {
    display: grid !important;
    gap: 12px !important;
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
}

.metric-strip .block {
    background: var(--od-surface-warm) !important;
}

.status-pill {
    align-items: center;
    background: var(--od-surface);
    border-radius: var(--od-radius-pill);
    color: var(--od-muted);
    display: inline-flex;
    font-size: 12px;
    font-weight: 500;
    gap: 6px;
    line-height: 1;
    padding: 5px 10px;
}

.status-pill[data-tone="active"],
.status-pill-active {
    background: rgba(0, 112, 243, 0.09);
    color: var(--od-accent-hover);
}

.code-surface {
    background: var(--od-surface-warm) !important;
    border: 1px solid var(--od-border-soft) !important;
    border-radius: var(--od-radius-md) !important;
    color: var(--od-fg) !important;
    font-family: var(--od-font-mono) !important;
    padding: 14px !important;
}

.workbench-empty-state {
    align-items: center;
    background: var(--od-surface-warm);
    border: 1px dashed var(--od-border);
    border-radius: var(--od-radius-md);
    color: var(--od-muted);
    display: flex;
    font-size: 13px;
    justify-content: center;
    min-height: 180px;
    padding: 20px;
    text-align: center;
}

.workbench-legacy-page-shell {
    margin-top: 0 !important;
}

.workbench-legacy-page-context {
    background: var(--od-surface-warm) !important;
}

.workbench-legacy-page-context-list {
    display: grid;
    gap: 12px;
    margin: 0;
}

.workbench-legacy-page-context-list div {
    border-top: 1px solid var(--od-border-soft);
    padding-top: 10px;
}

.workbench-legacy-page-context-list div:first-child {
    border-top: 0;
    padding-top: 0;
}

.workbench-legacy-page-context-list dt {
    color: var(--od-meta);
    font-size: 11px;
    font-weight: 600;
    line-height: 1;
    margin: 0 0 5px;
    text-transform: uppercase;
}

.workbench-legacy-page-context-list dd {
    color: var(--od-fg);
    font-size: 13px;
    line-height: 1.45;
    margin: 0;
    overflow-wrap: anywhere;
}

.workbench-legacy-output-panel {
    overflow: hidden !important;
}

.workbench-legacy-output-panel > .form,
.workbench-legacy-output-panel > .group {
    border-color: transparent !important;
}

/* Typography */
.prose,
.markdown,
.gr-prose,
.gradio-container p,
.gradio-container label {
    color: var(--od-fg);
    line-height: 1.6 !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4 {
    color: var(--od-fg) !important;
    font-weight: 600 !important;
}

.gradio-container code,
.gradio-container pre,
.token-display,
.token,
textarea,
input {
    font-family: var(--od-font-mono) !important;
}

.prose code,
.markdown code,
.gr-prose code {
    background: var(--od-surface) !important;
    border: 1px solid var(--od-border-soft) !important;
    color: var(--od-fg) !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
}

/* Layout and blocks */
.block,
.panel,
.form,
.group,
.accordion,
fieldset {
    border-color: var(--od-border-soft) !important;
    border-radius: var(--od-radius-md) !important;
    background: var(--od-bg) !important;
    box-shadow: none !important;
}

.form,
.group {
    border: 1px solid var(--od-border-soft) !important;
}

.block_label,
.block-title,
[data-testid="block-info"],
label,
.wrap label {
    background: transparent !important;
    color: var(--od-muted) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 0 !important;
    border-radius: 0 !important;
}

input[type="checkbox"],
input[type="radio"] {
    accent-color: var(--od-accent) !important;
}

label.selected,
.wrap label.selected {
    background: rgba(23, 23, 23, 0.06) !important;
    border-color: rgba(23, 23, 23, 0.28) !important;
    color: var(--od-fg) !important;
    border-radius: var(--od-radius-pill) !important;
}

label.selected span,
.wrap label.selected span {
    color: var(--od-fg) !important;
}

.info-panel {
    background: var(--od-surface-warm) !important;
    border: 1px solid var(--od-border-soft) !important;
    border-radius: var(--od-radius-md) !important;
    padding: 16px !important;
}

.module-title {
    border-bottom: 1px solid var(--od-border) !important;
    color: var(--od-fg) !important;
    font-size: 24px !important;
    font-weight: 600 !important;
    padding-bottom: 12px !important;
    margin-bottom: 20px !important;
}

/* Tabs */
.tabs,
.tabitem {
    background: var(--od-bg) !important;
}

.tab-nav {
    gap: 4px !important;
    border-bottom: 1px solid var(--od-border) !important;
    background: var(--od-bg) !important;
}

.tab-nav button {
    border: 1px solid transparent !important;
    border-radius: var(--od-radius-sm) var(--od-radius-sm) 0 0 !important;
    color: var(--od-muted) !important;
    background: transparent !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 9px 12px !important;
    transition: color 150ms var(--od-ease), background-color 150ms var(--od-ease), border-color 150ms var(--od-ease) !important;
}

.tab-nav button:hover {
    color: var(--od-fg) !important;
    background: var(--od-surface-warm) !important;
}

.tab-nav button.selected {
    color: var(--od-fg) !important;
    background: var(--od-bg) !important;
    border-color: var(--od-border) !important;
    border-bottom-color: var(--od-bg) !important;
}

/* Inputs */
input,
textarea,
select,
.input,
.textbox,
.dropdown {
    background: var(--od-bg) !important;
    border-color: var(--od-border) !important;
    color: var(--od-fg) !important;
    border-radius: var(--od-radius-sm) !important;
    box-shadow: none !important;
}

input:focus,
textarea:focus,
select:focus,
.input:focus,
.textbox:focus,
.dropdown:focus-within {
    border-color: var(--od-accent) !important;
    box-shadow: var(--od-focus-ring) !important;
    outline: none !important;
}

input::placeholder,
textarea::placeholder {
    color: var(--od-meta) !important;
}

/* Buttons */
button,
.button {
    border-radius: var(--od-radius-sm) !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    transition: background-color 150ms var(--od-ease), border-color 150ms var(--od-ease), color 150ms var(--od-ease) !important;
}

button.primary,
button[variant="primary"],
.primary-btn {
    background: var(--od-fg) !important;
    border-color: var(--od-fg) !important;
    color: #ffffff !important;
}

button.primary:hover,
button[variant="primary"]:hover,
.primary-btn:hover {
    background: var(--od-fg-2) !important;
    border-color: var(--od-fg-2) !important;
    transform: none !important;
}

.preset-btn,
button.secondary,
button[variant="secondary"] {
    background: var(--od-bg) !important;
    border: 1px solid var(--od-border) !important;
    color: var(--od-fg) !important;
}

.preset-btn:hover,
button.secondary:hover,
button[variant="secondary"]:hover {
    background: var(--od-surface-warm) !important;
    border-color: #d4d4d4 !important;
}

/* Status and cards */
.stat-card {
    background: var(--od-surface-warm) !important;
    border: 1px solid var(--od-border-soft) !important;
    border-radius: var(--od-radius-md) !important;
    padding: 16px !important;
    text-align: left !important;
}

.stat-value {
    color: var(--od-fg) !important;
    font-family: var(--od-font-body) !important;
    font-size: 24px !important;
    font-weight: 600 !important;
}

.stat-label,
.stat-unit {
    color: var(--od-muted) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: none !important;
}

.tip-box,
.warning-box {
    background: var(--od-surface-warm) !important;
    border: 1px solid var(--od-border-soft) !important;
    border-left: 3px solid var(--od-accent) !important;
    border-radius: var(--od-radius-sm) !important;
    padding: 12px 14px !important;
}

.warning-box {
    border-left-color: var(--od-warning) !important;
}

/* Token visualization */
.token-display {
    background: var(--od-surface-warm) !important;
    border: 1px solid var(--od-border-soft) !important;
    border-radius: var(--od-radius-md) !important;
    color: var(--od-fg) !important;
    padding: 16px !important;
    line-height: 2.25 !important;
}

.token {
    border-radius: 8px !important;
    border: 1px solid color-mix(in oklab, var(--od-accent), white 70%) !important;
    color: var(--od-fg) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 4px 8px !important;
    transition: border-color 150ms var(--od-ease), background-color 150ms var(--od-ease) !important;
}

.token:hover {
    transform: none !important;
    box-shadow: none !important;
    border-color: var(--od-accent) !important;
}

.token-special {
    border: 1px solid var(--od-danger) !important;
}

.token-byte {
    border: 1px dashed var(--od-warning) !important;
}

.token-color-0 { background-color: #e8f5f0; }
.token-color-1 { background-color: #eef2ff; }
.token-color-2 { background-color: #f5f5f5; }
.token-color-3 { background-color: #fff7ed; }
.token-color-4 { background-color: #fef2f2; }
.token-color-5 { background-color: #eff6ff; }
.token-color-6 { background-color: #f0fdf4; }
.token-color-7 { background-color: #faf5ff; }
.token-color-8 { background-color: #fefce8; }
.token-color-9 { background-color: #f8fafc; }
.token-color-10 { background-color: #fdf2f8; }
.token-color-11 { background-color: #ecfeff; }

/* Tables and dataframes */
table {
    border-collapse: collapse !important;
}

table th,
table td,
.prose th,
.prose td {
    border-color: var(--od-border) !important;
}

.dataframe,
.table-wrap {
    border: 1px solid var(--od-border-soft) !important;
    border-radius: var(--od-radius-md) !important;
    overflow: hidden !important;
    font-size: 13px !important;
}

.dataframe th,
.dataframe thead {
    background: var(--od-surface-warm) !important;
    color: var(--od-muted) !important;
    font-weight: 500 !important;
}

/* Plot containers */
.plot,
.plot-container,
.js-plotly-plot {
    border-radius: var(--od-radius-md) !important;
}

.plot-frame .plot,
.plot-frame .plot-container,
.plot-frame .js-plotly-plot {
    background: var(--od-bg) !important;
}

@media (max-width: 760px) {
    .workbench-header {
        align-items: flex-start;
        flex-direction: column;
        padding: 12px 16px;
    }

    .workbench-header-actions {
        justify-content: flex-start;
    }

    .workbench-page {
        padding: 18px 16px 28px;
    }

    .workbench-app-layout {
        padding: 12px !important;
    }

    .workbench-command-surface {
        gap: 12px !important;
        padding: 12px !important;
    }

    .workbench-command-header {
        align-items: start;
        display: flex;
        flex-direction: column;
    }

    .workbench-launcher-controls {
        grid-template-columns: 1fr !important;
    }

    .workbench-command-surface .workbench-launcher-controls > .form {
        grid-template-columns: 1fr !important;
    }

    .workbench-page-switcher {
        display: flex !important;
        flex-direction: column !important;
    }

    .workbench-page-switcher > .tabitem {
        width: 100% !important;
    }

    .workbench-tool-shell {
        display: flex !important;
        flex-direction: column !important;
        gap: 10px !important;
    }

    .workbench-control-panel {
        position: static !important;
    }

    .metric-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }

    .workbench-page-hero,
    .main-header {
        padding: 2px 0 16px !important;
    }
}
"""


CUSTOM_THEME = gr.themes.Soft(
    primary_hue="gray",
    secondary_hue="gray",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    radius_size="md",
    spacing_size="md",
).set(
    body_background_fill=OPEN_DESIGN_COLORS["bg"],
    body_text_color=OPEN_DESIGN_COLORS["fg"],
    block_background_fill=OPEN_DESIGN_COLORS["bg"],
    block_border_color=OPEN_DESIGN_COLORS["border_soft"],
    block_border_width="1px",
    block_radius="12px",
    block_shadow="none",
    button_primary_background_fill=OPEN_DESIGN_COLORS["fg"],
    button_primary_background_fill_hover=OPEN_DESIGN_COLORS["fg_2"],
    button_primary_border_color=OPEN_DESIGN_COLORS["fg"],
    button_primary_text_color="#ffffff",
    button_secondary_background_fill=OPEN_DESIGN_COLORS["bg"],
    button_secondary_background_fill_hover=OPEN_DESIGN_COLORS["surface_warm"],
    button_secondary_border_color=OPEN_DESIGN_COLORS["border"],
    button_secondary_text_color=OPEN_DESIGN_COLORS["fg"],
    input_background_fill=OPEN_DESIGN_COLORS["bg"],
    input_border_color=OPEN_DESIGN_COLORS["border"],
    input_border_color_focus=OPEN_DESIGN_COLORS["accent"],
    input_shadow_focus="0 0 0 3px rgba(0, 112, 243, 0.14)",
    checkbox_background_color_selected=OPEN_DESIGN_COLORS["accent"],
    checkbox_border_color_selected=OPEN_DESIGN_COLORS["accent"],
    slider_color=OPEN_DESIGN_COLORS["accent"],
    table_border_color=OPEN_DESIGN_COLORS["border"],
)


def render_app_header() -> str:
    """渲染应用顶部品牌区。"""
    return """
    <header class="workbench-header">
      <div class="workbench-brand">
        <div class="workbench-mark" aria-hidden="true"></div>
        <div>
          <h1 class="workbench-title">LLM Tools Workbench</h1>
          <p class="workbench-subtitle">Dense local workbench for inspecting language model mechanics.</p>
        </div>
      </div>
      <div class="workbench-header-actions">
        <span class="workbench-pill">Local-first</span>
        <span class="workbench-pill">Gradio</span>
        <span class="workbench-pill">Awesome DESIGN.md</span>
      </div>
    </header>
    """


def render_command_header(pages: list[dict[str, str]]) -> str:
    """渲染紧凑工具启动区标题。"""
    tool_count = len(pages)

    return f"""
    <section class="workbench-command-header" aria-label="Workbench command header">
      <div>
        <h2 class="workbench-command-title">Open a Work Surface</h2>
        <p class="workbench-command-copy">Pick a task area, then choose one focused tool. No summary card stack, no long tool rail.</p>
      </div>
      <div class="workbench-command-meta">
        <span class="workbench-pill">{escape(str(tool_count))} tools</span>
        <span class="workbench-pill">Vercel / Linear density</span>
      </div>
    </section>
    """


def render_legacy_page_header(page: dict[str, str], page_name: str) -> str:
    """渲染未迁移页面的统一标题区。"""
    title = page_name
    description = f'{page["group_description"]}'
    meta = f'{page["group"]} / {page["lab"]}'

    return f"""
    <section class="workbench-page-hero">
      <h1>{escape(title)}</h1>
      <p>{escape(meta)}. {escape(description)}</p>
    </section>
    """


def render_legacy_page_context(page: dict[str, str], page_name: str) -> str:
    """渲染未迁移页面的统一上下文栏。"""
    return f"""
    <div>
      <div class="workbench-panel-title">Page Context</div>
      <dl class="workbench-legacy-page-context-list">
        <div>
          <dt>Section</dt>
          <dd>{escape(page["group"])}</dd>
        </div>
        <div>
          <dt>Lab</dt>
          <dd>{escape(page["lab"])}</dd>
        </div>
        <div>
          <dt>Tool</dt>
          <dd>{escape(page_name)}</dd>
        </div>
      </dl>
    </div>
    """


def render_app_footer() -> str:
    """闭合应用主内容容器。"""
    return ""


def configure_plotly_theme() -> None:
    """设置 Plotly 默认视觉，使各页面图表靠近 Open Design token。"""
    template = {
        "layout": {
            "font": {
                "family": "Inter, system-ui, -apple-system, Segoe UI, sans-serif",
                "color": OPEN_DESIGN_COLORS["fg"],
                "size": 13,
            },
            "paper_bgcolor": OPEN_DESIGN_COLORS["bg"],
            "plot_bgcolor": OPEN_DESIGN_COLORS["bg"],
            "colorway": PLOTLY_COLORWAY,
            "margin": {"l": 56, "r": 32, "t": 56, "b": 56},
            "xaxis": {
                "gridcolor": OPEN_DESIGN_COLORS["border_soft"],
                "linecolor": OPEN_DESIGN_COLORS["border"],
                "zerolinecolor": OPEN_DESIGN_COLORS["border"],
                "title": {"font": {"color": OPEN_DESIGN_COLORS["muted"]}},
                "tickfont": {"color": OPEN_DESIGN_COLORS["muted"]},
            },
            "yaxis": {
                "gridcolor": OPEN_DESIGN_COLORS["border_soft"],
                "linecolor": OPEN_DESIGN_COLORS["border"],
                "zerolinecolor": OPEN_DESIGN_COLORS["border"],
                "title": {"font": {"color": OPEN_DESIGN_COLORS["muted"]}},
                "tickfont": {"color": OPEN_DESIGN_COLORS["muted"]},
            },
            "legend": {
                "font": {"color": OPEN_DESIGN_COLORS["muted"]},
                "bgcolor": "rgba(255,255,255,0)",
            },
        }
    }
    pio.templates["workbench_open_design"] = template
    pio.templates.default = "plotly_white+workbench_open_design"
