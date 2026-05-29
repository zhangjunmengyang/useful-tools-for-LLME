"""
LLM Tools Workbench - Gradio 版本。

一个用于大模型学习与实验的可视化工具集。
"""

import importlib
import importlib.util
from collections.abc import Callable
from typing import Any

import gradio as gr

from workbench_theme import (
    CUSTOM_CSS,
    CUSTOM_THEME,
    configure_plotly_theme,
    render_app_footer,
    render_app_header,
    render_command_header,
    render_legacy_page_context,
    render_legacy_page_header,
)


PAGE_REGISTRY: list[dict[str, str]] = [
    {
        "id": "toolbox_tool_runner",
        "group": "Research Toolbox",
        "group_description": "Run reusable tools and export structured research artifacts.",
        "lab": "Toolbox",
        "label": "Tool Runner",
        "module": "toolbox_lab.tool_runner",
    },
    {
        "id": "token_playground",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "TokenLab",
        "label": "Playground",
        "module": "token_lab.playground",
    },
    {
        "id": "token_arena",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "TokenLab",
        "label": "Arena",
        "module": "token_lab.arena",
    },
    {
        "id": "token_chat_template",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "TokenLab",
        "label": "Chat Template",
        "module": "token_lab.chat_builder",
    },
    {
        "id": "embedding_vector_arithmetic",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "EmbeddingLab",
        "label": "Vector Arithmetic",
        "module": "embedding_lab.vector_arithmetic",
    },
    {
        "id": "embedding_model_comparison",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "EmbeddingLab",
        "label": "Model Comparison",
        "module": "embedding_lab.model_comparison",
    },
    {
        "id": "embedding_visualization",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "EmbeddingLab",
        "label": "Visualization",
        "module": "embedding_lab.vector_visualization",
    },
    {
        "id": "embedding_semantic_similarity",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "EmbeddingLab",
        "label": "Semantic Similarity",
        "module": "embedding_lab.semantic_similarity",
    },
    {
        "id": "generation_logits",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "GenerationLab",
        "label": "Logits Inspector",
        "module": "generation_lab.logits_inspector",
    },
    {
        "id": "generation_beam",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "GenerationLab",
        "label": "Beam Search",
        "module": "generation_lab.beam_visualizer",
    },
    {
        "id": "generation_kv_cache",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "GenerationLab",
        "label": "KV Cache",
        "module": "generation_lab.kv_cache_sim",
    },
    {
        "id": "interpretability_attention",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "InterpretabilityLab",
        "label": "Attention",
        "module": "interpretability_lab.attention_map",
    },
    {
        "id": "interpretability_rope",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "InterpretabilityLab",
        "label": "RoPE Explorer",
        "module": "interpretability_lab.rope_explorer",
    },
    {
        "id": "interpretability_ffn",
        "group": "Core Mechanics",
        "group_description": "Inspect tokens, vectors, generation, and model internals.",
        "lab": "InterpretabilityLab",
        "label": "FFN Activation",
        "module": "interpretability_lab.ffn_activation",
    },
    {
        "id": "data_dataset_viewer",
        "group": "Knowledge & Data",
        "group_description": "Prepare datasets and reason about retrieval behavior.",
        "lab": "DataLab",
        "label": "Dataset Viewer",
        "module": "data_lab.hf_dataset_viewer",
    },
    {
        "id": "data_cleaner",
        "group": "Knowledge & Data",
        "group_description": "Prepare datasets and reason about retrieval behavior.",
        "lab": "DataLab",
        "label": "Data Cleaner",
        "module": "data_lab.cleaner_playground",
    },
    {
        "id": "data_formatter",
        "group": "Knowledge & Data",
        "group_description": "Prepare datasets and reason about retrieval behavior.",
        "lab": "DataLab",
        "label": "Format Converter",
        "module": "data_lab.instruct_formatter",
    },
    {
        "id": "rag_chunking",
        "group": "Knowledge & Data",
        "group_description": "Prepare datasets and reason about retrieval behavior.",
        "lab": "RAGLab",
        "label": "Chunking",
        "module": "rag_lab.chunking_playground",
    },
    {
        "id": "rag_retrieval",
        "group": "Knowledge & Data",
        "group_description": "Prepare datasets and reason about retrieval behavior.",
        "lab": "RAGLab",
        "label": "Retrieval",
        "module": "rag_lab.retrieval_sim",
    },
    {
        "id": "model_memory",
        "group": "Model Ops",
        "group_description": "Estimate memory, fine-tuning cost, and inference constraints.",
        "lab": "ModelLab",
        "label": "Memory Estimator",
        "module": "model_lab.memory_estimator",
    },
    {
        "id": "model_peft",
        "group": "Model Ops",
        "group_description": "Estimate memory, fine-tuning cost, and inference constraints.",
        "lab": "ModelLab",
        "label": "PEFT Calculator",
        "module": "model_lab.peft_calculator",
    },
    {
        "id": "model_config_diff",
        "group": "Model Ops",
        "group_description": "Estimate memory, fine-tuning cost, and inference constraints.",
        "lab": "ModelLab",
        "label": "Config Diff",
        "module": "model_lab.config_diff",
    },
    {
        "id": "finetune_lora",
        "group": "Model Ops",
        "group_description": "Estimate memory, fine-tuning cost, and inference constraints.",
        "lab": "FineTuneLab",
        "label": "LoRA Explorer",
        "module": "finetune_lab.lora_explorer",
    },
    {
        "id": "finetune_training_cost",
        "group": "Model Ops",
        "group_description": "Estimate memory, fine-tuning cost, and inference constraints.",
        "lab": "FineTuneLab",
        "label": "Training Cost",
        "module": "finetune_lab.training_cost_estimator",
    },
    {
        "id": "inference_throughput",
        "group": "Model Ops",
        "group_description": "Estimate memory, fine-tuning cost, and inference constraints.",
        "lab": "InferenceLab",
        "label": "Throughput",
        "module": "inference_lab.throughput_calculator",
        "optional_package": "inference_lab",
    },
    {
        "id": "inference_quantization",
        "group": "Model Ops",
        "group_description": "Estimate memory, fine-tuning cost, and inference constraints.",
        "lab": "InferenceLab",
        "label": "Quantization",
        "module": "inference_lab.quantization_analyzer",
        "optional_package": "inference_lab",
    },
    {
        "id": "agent_trace_viewer",
        "group": "Evaluation",
        "group_description": "Review agent traces and compare evaluation pipelines.",
        "lab": "Agent Trace Lab",
        "label": "Trace Viewer",
        "module": "agent_trace_lab.trace_viewer",
    },
    {
        "id": "agent_trace_analyzer",
        "group": "Evaluation",
        "group_description": "Review agent traces and compare evaluation pipelines.",
        "lab": "Agent Trace Lab",
        "label": "Trace Analyzer",
        "module": "agent_trace_lab.trace_analyzer",
    },
    {
        "id": "eval_benchmark",
        "group": "Evaluation",
        "group_description": "Review agent traces and compare evaluation pipelines.",
        "lab": "Eval Lab",
        "label": "Benchmark Explorer",
        "module": "eval_lab.benchmark_explorer",
    },
    {
        "id": "eval_llm_judge",
        "group": "Evaluation",
        "group_description": "Review agent traces and compare evaluation pipelines.",
        "lab": "Eval Lab",
        "label": "LLM Judge",
        "module": "eval_lab.llm_judge",
    },
    {
        "id": "eval_pipeline",
        "group": "Evaluation",
        "group_description": "Review agent traces and compare evaluation pipelines.",
        "lab": "Eval Lab",
        "label": "Eval Pipeline",
        "module": "eval_lab.eval_pipeline",
    },
]

LAB_NAV_LABELS = {
    "TokenLab": "Token",
    "EmbeddingLab": "Embedding",
    "GenerationLab": "Generation",
    "InterpretabilityLab": "Interpretability",
    "DataLab": "Data",
    "RAGLab": "RAG",
    "ModelLab": "Model",
    "FineTuneLab": "FineTune",
    "InferenceLab": "Inference",
    "Agent Trace Lab": "Agent Trace",
    "Eval Lab": "Eval",
    "Toolbox": "Toolbox",
}

WORKBENCH_LAYOUT_MARKERS = (
    "workbench-page-hero",
    "workbench-tool-shell",
    "workbench-control-panel",
    "workbench-output-panel",
)


def optional_lab_available(package_name: str) -> bool:
    """判断可选实验室模块是否存在。"""
    return importlib.util.find_spec(package_name) is not None


def get_available_pages() -> list[dict[str, str]]:
    """返回当前环境可用的页面注册表。"""
    pages = []
    for page in PAGE_REGISTRY:
        optional_package = page.get("optional_package")
        if optional_package and not optional_lab_available(optional_package):
            continue
        pages.append(page)
    return pages


def _empty_outputs(output_count: int) -> Any:
    """生成与 Gradio 输出数量匹配的空值。"""
    if output_count == 1:
        return None
    return tuple(None for _ in range(output_count))


def _cached_load_handler(result: dict[str, Any], page_name: str) -> Callable[[], Any]:
    """为 Tab 懒加载创建单次缓存包装。"""
    load_fn = result["load_fn"]
    output_count = len(result["load_outputs"])
    cache: dict[str, Any] = {"ready": False, "value": None}

    def load_once() -> Any:
        if cache["ready"]:
            return cache["value"]

        try:
            value = load_fn()
        except Exception as exc:
            print(f"Load event error in {page_name}: {exc}")
            value = _empty_outputs(output_count)

        cache["value"] = value
        cache["ready"] = True
        return value

    return load_once


def bind_lazy_load_events(tabs: list[gr.Tab], result: Any, page_name: str) -> None:
    """把页面初始化绑定到一个或多个 Tab，避免应用启动时预加载所有模型。"""
    if not result:
        return

    load_handler = _cached_load_handler(result, page_name)
    for tab in tabs:
        tab.select(fn=load_handler, outputs=result["load_outputs"])


def bind_lazy_load_event(tab: gr.Tab, result: Any, page_name: str) -> None:
    """把页面初始化绑定到对应 Tab。"""
    bind_lazy_load_events([tab], result, page_name)


def _render_page(page: dict[str, str]) -> Any:
    """导入并渲染一个页面模块。"""
    module = importlib.import_module(page["module"])
    return module.render()


def _page_source(page: dict[str, str]) -> str:
    """读取页面模块源码，用于识别是否已迁移到 workbench 布局。"""
    try:
        spec = importlib.util.find_spec(page["module"])
    except (ModuleNotFoundError, ValueError):
        return ""

    if spec is None or spec.origin is None:
        return ""

    try:
        with open(spec.origin, encoding="utf-8") as source_file:
            return source_file.read()
    except OSError:
        return ""


def _page_uses_workbench_layout(page: dict[str, str]) -> bool:
    """判断页面是否已经原生使用 workbench 布局。"""
    source = _page_source(page)
    return all(marker in source for marker in WORKBENCH_LAYOUT_MARKERS)


def _render_page_with_frame(page: dict[str, str], page_name: str) -> Any:
    """渲染页面；未迁移页面自动套入统一 workbench frame。"""
    if _page_uses_workbench_layout(page):
        return _render_page(page)

    gr.HTML(render_legacy_page_header(page, page_name))
    with gr.Row(elem_classes=["workbench-tool-shell", "workbench-legacy-page-shell"]):
        with gr.Column(
            scale=1,
            elem_classes=["workbench-control-panel", "workbench-legacy-page-context"],
        ):
            gr.HTML(render_legacy_page_context(page, page_name))
        with gr.Column(
            scale=3,
            elem_classes=["workbench-output-panel", "workbench-legacy-output-panel"],
        ):
            result = _render_page(page)

    return result


def _page_tab_label(page: dict[str, str]) -> str:
    """生成工具选择器使用的紧凑页面标题。"""
    lab_label = LAB_NAV_LABELS.get(page["lab"], page["lab"])
    return f'{lab_label} / {page["label"]}'


def _page_groups(pages: list[dict[str, str]]) -> list[str]:
    """按注册顺序返回任务域列表。"""
    groups = []
    for page in pages:
        if page["group"] not in groups:
            groups.append(page["group"])
    return groups


def _tool_choices_for_group(
    pages: list[dict[str, str]],
    group: str,
) -> list[tuple[str, str]]:
    """返回某个任务域下的工具选择项。"""
    return [
        (_page_tab_label(page), page["id"])
        for page in pages
        if page["group"] == group
    ]


def _first_page_id_for_group(pages: list[dict[str, str]], group: str) -> str:
    """返回任务域中的第一个页面 id。"""
    for page in pages:
        if page["group"] == group:
            return page["id"]
    return pages[0]["id"]


def _group_for_page_id(pages: list[dict[str, str]], page_id: str) -> str:
    """返回页面 id 所属的任务域。"""
    for page in pages:
        if page["id"] == page_id:
            return page["group"]
    return pages[0]["group"]


def create_app():
    """创建 Gradio 应用。"""
    configure_plotly_theme()
    pages = get_available_pages()
    default_page_id = pages[0]["id"]
    default_group = pages[0]["group"]
    groups = _page_groups(pages)

    with gr.Blocks(
        title="LLM Tools Workbench",
        analytics_enabled=False,
    ) as app:
        gr.HTML(render_app_header())

        with gr.Column(elem_classes=["workbench-app-layout"]):
            with gr.Column(elem_classes=["workbench-command-surface"]):
                gr.HTML(render_command_header(pages))
                with gr.Row(elem_classes=["workbench-launcher-controls"]):
                    group_selector = gr.Dropdown(
                        label="Task Area",
                        choices=groups,
                        value=default_group,
                        interactive=True,
                        elem_classes=["workbench-group-selector"],
                    )
                    tool_selector = gr.Dropdown(
                        label="Tool",
                        choices=_tool_choices_for_group(pages, default_group),
                        value=default_page_id,
                        filterable=True,
                        interactive=True,
                        elem_classes=["workbench-tool-selector"],
                    )

            with gr.Tabs(
                selected=default_page_id,
                elem_classes=["workbench-page-switcher", "workbench-page-router"],
            ) as page_tabs:
                for page in pages:
                    page_name = _page_tab_label(page)
                    with gr.Tab(
                        page_name,
                        id=page["id"],
                    ) as page_tab:
                        result = _render_page_with_frame(page, page_name)
                        bind_lazy_load_event(
                            page_tab,
                            result,
                            page_name,
                        )

            def update_group(group: str) -> tuple[Any, Any]:
                """切换任务域，同时打开该域的第一个工具。"""
                page_id = _first_page_id_for_group(pages, group)
                return (
                    gr.update(
                        choices=_tool_choices_for_group(pages, group),
                        value=page_id,
                    ),
                    gr.Tabs(selected=page_id),
                )

            def update_tool(page_id: str) -> tuple[Any, Any, Any]:
                """切换当前工具页面，并保持任务域与工具状态一致。"""
                group = _group_for_page_id(pages, page_id)
                return (
                    gr.update(value=group),
                    gr.update(
                        choices=_tool_choices_for_group(pages, group),
                        value=page_id,
                    ),
                    gr.Tabs(selected=page_id),
                )

            group_selector.change(
                fn=update_group,
                inputs=[group_selector],
                outputs=[tool_selector, page_tabs],
            )
            tool_selector.change(
                fn=update_tool,
                inputs=[tool_selector],
                outputs=[group_selector, tool_selector, page_tabs],
            )

        gr.HTML(render_app_footer())

    return app


if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS,
    )
