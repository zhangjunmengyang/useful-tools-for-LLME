"""Generic research tool runner page."""

from __future__ import annotations

import json
from html import escape
from typing import Any

import gradio as gr

from workbench_tools.default_configs import DEFAULT_CONFIGS
from workbench_tools.registry import get_registry


def _all_specs():
    """返回所有工具定义。"""
    return get_registry().list_specs()


def _lab_choices() -> list[str]:
    """返回 Lab 过滤选项。"""
    labs = sorted({spec.lab for spec in _all_specs()})
    return ["All"] + labs


def _concept_choices() -> list[str]:
    """返回 concept 过滤选项。"""
    concepts = sorted({concept for spec in _all_specs() for concept in spec.concepts})
    return ["All"] + concepts


def _tool_choices(lab_filter: str = "All", concept_filter: str = "All") -> list[tuple[str, str]]:
    """生成工具下拉选择项。"""
    specs = _all_specs()
    if lab_filter and lab_filter != "All":
        specs = [spec for spec in specs if spec.lab == lab_filter]
    if concept_filter and concept_filter != "All":
        specs = [spec for spec in specs if concept_filter in spec.concepts]
    return [
        (f"{spec.label} ({spec.id})", spec.id)
        for spec in specs
    ]


def get_tool_catalog(lab_filter: str = "All", concept_filter: str = "All") -> list[list[str]]:
    """返回工具目录表格。"""
    specs = _all_specs()
    if lab_filter and lab_filter != "All":
        specs = [spec for spec in specs if spec.lab == lab_filter]
    if concept_filter and concept_filter != "All":
        specs = [spec for spec in specs if concept_filter in spec.concepts]
    return [
        [
            spec.id,
            spec.label,
            spec.lab,
            ", ".join(spec.concepts),
            "Yes" if spec.requires_model_download else "No",
        ]
        for spec in specs
    ]


def get_tool_catalog_markdown(lab_filter: str = "All", concept_filter: str = "All") -> str:
    """返回工具目录 Markdown 表格。"""
    rows = get_tool_catalog(lab_filter, concept_filter)
    if not rows:
        return "No matching tools."

    lines = [
        "| Tool ID | Label | Lab | Concepts | Downloads Model |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tool_id, label, lab, concepts, downloads_model in rows:
        lines.append(
            f"| `{tool_id}` | {label} | {lab} | {concepts or '-'} | {downloads_model} |"
        )
    return "\n".join(lines)


def _default_tool_id() -> str:
    """返回默认工具 id。"""
    specs = get_registry().list_specs()
    return specs[0].id if specs else ""


def _format_json(value: Any) -> str:
    """格式化 JSON 展示文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2)


def _render_tool_details_html(tool_id: str | None) -> str:
    """渲染当前工具的可扫读说明。"""
    if not tool_id:
        return """
        <section class="tool-runner-overview">
          <div>
            <p class="tool-runner-eyebrow">Selected Tool</p>
            <h2>No tool selected</h2>
            <p>Adjust the filters above to choose an available research tool.</p>
          </div>
        </section>
        """

    spec = get_registry().get_spec(tool_id)
    concepts = spec.concepts or []
    concept_chips = "\n".join(
        f'<span class="tool-runner-chip">{escape(concept)}</span>'
        for concept in concepts
    )
    model_badge = (
        "Model download required"
        if spec.requires_model_download
        else "No model download"
    )

    return f"""
    <section class="tool-runner-overview">
      <div class="tool-runner-overview-main">
        <p class="tool-runner-eyebrow">Selected Tool</p>
        <h2>{escape(spec.label)}</h2>
        <p>{escape(spec.description)}</p>
      </div>
      <div class="tool-runner-meta-grid">
        <div>
          <span>Tool ID</span>
          <code>{escape(spec.id)}</code>
        </div>
        <div>
          <span>Lab</span>
          <strong>{escape(spec.lab)}</strong>
        </div>
        <div>
          <span>Runtime</span>
          <strong>{escape(model_badge)}</strong>
        </div>
      </div>
      <div class="tool-runner-chip-row" aria-label="Tool concepts">
        {concept_chips or '<span class="tool-runner-chip">No concepts</span>'}
      </div>
    </section>
    """


def get_tool_details(tool_id: str) -> tuple[str, str, str, str]:
    """返回工具说明和默认配置。"""
    if not tool_id:
        return _render_tool_details_html(None), "{}", "{}", ""

    spec = get_registry().get_spec(tool_id)
    cli_command = f"python -m workbench_tools run {spec.id} --config config.json --output-dir research"
    return (
        _render_tool_details_html(tool_id),
        _format_json(DEFAULT_CONFIGS.get(tool_id, {})),
        _format_json(spec.input_schema),
        cli_command,
    )


def update_tool_choices(lab_filter: str, concept_filter: str) -> tuple[Any, str, str, str, str, str]:
    """根据过滤条件刷新工具列表。"""
    choices = _tool_choices(lab_filter, concept_filter)
    selected = choices[0][1] if choices else None
    details, config, schema, command = (
        get_tool_details(selected)
        if selected
        else ("No matching tool.", "{}", "{}", "")
    )
    return (
        gr.update(choices=choices, value=selected),
        details,
        config,
        schema,
        command,
        get_tool_catalog_markdown(lab_filter, concept_filter),
    )


def run_selected_tool(
    tool_id: str,
    config_text: str,
    export_artifact: bool,
    output_dir: str,
) -> tuple[str, str, str]:
    """执行工具并返回 UI 展示结果。"""
    try:
        inputs = json.loads(config_text or "{}")
    except json.JSONDecodeError as exc:
        payload = {
            "status": "error",
            "error": f"Invalid JSON config: {exc}",
        }
        return "Invalid JSON config", _format_json(payload), ""

    run = get_registry().run(
        tool_id,
        inputs,
        export=export_artifact,
        output_dir=output_dir or "research",
    )
    artifact_text = ""
    if run.artifact:
        artifact_text = (
            f"Markdown: `{run.artifact.markdown_path}`\n\n"
            f"JSON: `{run.artifact.json_path}`"
        )
    status = "Run complete" if run.status == "success" else "Run failed"
    return status, _format_json(run.to_dict()), artifact_text


def render():
    """渲染研究工具运行页面。"""
    default_tool = _default_tool_id()
    details, default_config, default_schema, default_command = get_tool_details(default_tool)

    with gr.Column(elem_classes=["workbench-tool-runner"]):
        gr.HTML(
            """
            <div class="workbench-page-hero">
              <h1>Research Tool Runner</h1>
              <p>Run reusable LLM research tools, inspect structured outputs, and export Markdown/JSON artifacts.</p>
            </div>
            """
        )

        with gr.Column(elem_classes=["workbench-tool-runner-toolbar"]):
            with gr.Row(elem_classes=["tool-runner-toolbar-row"]):
                lab_filter = gr.Dropdown(
                    label="Lab Filter",
                    choices=_lab_choices(),
                    value="All",
                )
                concept_filter = gr.Dropdown(
                    label="Concept Filter",
                    choices=_concept_choices(),
                    value="All",
                )
                tool_selector = gr.Dropdown(
                    label="Tool",
                    choices=_tool_choices(),
                    value=default_tool,
                    filterable=True,
                )

        with gr.Row(elem_classes=["workbench-tool-shell", "workbench-tool-runner-grid"]):
            with gr.Column(
                scale=1,
                elem_classes=["workbench-control-panel", "tool-runner-config-panel"],
            ):
                gr.Markdown("### Configure Run")
                gr.Markdown("Edit the JSON payload, choose export behavior, then run the selected tool.")
                config_input = gr.Code(
                    label="Input Config",
                    value=default_config,
                    language="json",
                    lines=18,
                )
                export_toggle = gr.Checkbox(
                    label="Export Markdown and JSON",
                    value=True,
                )
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="research",
                    placeholder="research",
                )
                run_button = gr.Button(
                    "Run Tool",
                    variant="primary",
                    elem_classes=["tool-runner-run-button"],
                )

            with gr.Column(
                scale=2,
                elem_classes=["workbench-output-panel", "tool-runner-results-panel"],
            ):
                tool_details = gr.HTML(value=details)

                with gr.Accordion("Implementation Details", open=False):
                    input_schema = gr.Code(
                        label="Input Schema",
                        value=default_schema,
                        language="json",
                        lines=12,
                    )
                    cli_command = gr.Textbox(
                        label="CLI Command",
                        value=default_command,
                        interactive=False,
                    )

                gr.Markdown("### Run Output")
                status = gr.Textbox(label="Status", interactive=False)
                with gr.Tabs(elem_classes=["tool-runner-result-tabs"]):
                    with gr.Tab("Tool Run JSON"):
                        result_json = gr.Code(
                            label="Tool Run JSON",
                            language="json",
                            lines=18,
                        )
                    with gr.Tab("Artifacts"):
                        artifact_paths = gr.Markdown(label="Artifacts")

                with gr.Accordion("Tool Catalog", open=False):
                    catalog_table = gr.Markdown(value=get_tool_catalog_markdown())

    lab_filter.change(
        fn=update_tool_choices,
        inputs=[lab_filter, concept_filter],
        outputs=[tool_selector, tool_details, config_input, input_schema, cli_command, catalog_table],
    )
    concept_filter.change(
        fn=update_tool_choices,
        inputs=[lab_filter, concept_filter],
        outputs=[tool_selector, tool_details, config_input, input_schema, cli_command, catalog_table],
    )
    tool_selector.change(
        fn=get_tool_details,
        inputs=[tool_selector],
        outputs=[tool_details, config_input, input_schema, cli_command],
    )
    run_button.click(
        fn=run_selected_tool,
        inputs=[tool_selector, config_input, export_toggle, output_dir],
        outputs=[status, result_json, artifact_paths],
    )
