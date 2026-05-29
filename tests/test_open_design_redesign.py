"""Awesome DESIGN.md redesign 回归测试。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OpenDesignRedesignTest(unittest.TestCase):
    """确保 Gradio 入口采用工具型 DESIGN.md 约束。"""

    def test_app_uses_shared_open_design_theme(self):
        app_source = (ROOT / "app_gradio.py").read_text(encoding="utf-8")

        self.assertIn("from workbench_theme import", app_source)
        self.assertIn("configure_plotly_theme()", app_source)
        self.assertIn("render_app_header()", app_source)
        self.assertIn("theme=CUSTOM_THEME", app_source)
        self.assertIn("css=CUSTOM_CSS", app_source)

    def test_theme_file_is_source_backed_by_awesome_design_tokens(self):
        theme_path = ROOT / "workbench_theme.py"
        self.assertTrue(theme_path.exists(), "workbench_theme.py should own shared UI tokens")

        theme_source = theme_path.read_text(encoding="utf-8")
        self.assertIn("VoltAgent/awesome-design-md", theme_source)
        self.assertIn("design-md/vercel/DESIGN.md", theme_source)
        self.assertIn("design-md/linear.app/DESIGN.md", theme_source)
        self.assertIn("design-md/raycast/DESIGN.md", theme_source)
        self.assertIn("#171717", theme_source)
        self.assertIn("#0070f3", theme_source)
        self.assertNotIn("#10a37f", theme_source)
        self.assertNotIn("#FF9D00", theme_source)
        self.assertNotIn("HuggingFace", theme_source)

    def test_theme_exposes_open_design_workbench_primitives(self):
        theme_source = (ROOT / "workbench_theme.py").read_text(encoding="utf-8")

        required_primitives = [
            "LAB_GROUPS",
            "render_command_header",
            ".workbench-app-layout",
            ".workbench-command-surface",
            ".workbench-command-header",
            ".workbench-launcher-controls",
            ".workbench-group-selector",
            ".workbench-tool-selector",
            ".workbench-page-router > .tab-wrapper",
            ".workbench-page-switcher > .tabitem .tab-wrapper",
            ".workbench-layout",
            ".workbench-tool-shell",
            ".workbench-control-panel",
            ".workbench-output-panel",
            ".workbench-detail-panel",
            ".metric-strip",
            ".status-pill",
            ".code-surface",
            ".workbench-empty-state",
            ".plot-frame",
            "--od-nav-width",
        ]
        for primitive in required_primitives:
            with self.subTest(primitive=primitive):
                self.assertIn(primitive, theme_source)

    def test_plotly_colors_are_restrained_to_open_design_tokens(self):
        theme_source = (ROOT / "workbench_theme.py").read_text(encoding="utf-8")

        self.assertIn('"stone": "#a1a1a1"', theme_source)
        self.assertNotIn("#7c3aed", theme_source)
        self.assertNotIn("#64748b", theme_source)

    def test_app_renders_grouped_lab_navigation(self):
        app_source = (ROOT / "app_gradio.py").read_text(encoding="utf-8")

        self.assertIn("PAGE_REGISTRY", app_source)
        self.assertIn("render_command_header", app_source)
        self.assertIn('elem_classes=["workbench-command-surface"]', app_source)
        self.assertIn("group_selector = gr.Dropdown", app_source)
        self.assertIn("interactive=True", app_source)
        self.assertIn("tool_selector = gr.Dropdown", app_source)
        self.assertIn("gr.Tabs(selected=page_id)", app_source)
        self.assertIn("_group_for_page_id", app_source)
        self.assertIn("outputs=[group_selector, tool_selector, page_tabs]", app_source)
        self.assertIn('elem_classes=["workbench-page-switcher", "workbench-page-router"]', app_source)
        self.assertIn("_page_tab_label", app_source)
        self.assertNotIn("_nav_updates", app_source)
        self.assertNotIn("render_page_navigation", app_source)
        self.assertNotIn("render_launcher_summary", app_source)
        self.assertNotIn("render_lab_group_rail", app_source)
        self.assertNotIn('with gr.Tabs():\n                # ==================== TokenLab', app_source)

    def test_tool_launcher_avoids_continuous_scroll_navigation(self):
        theme_source = (ROOT / "workbench_theme.py").read_text(encoding="utf-8")

        self.assertIn("workbench-command-surface", theme_source)
        self.assertIn("grid-template-columns: minmax(180px, 0.42fr) minmax(320px, 1fr)", theme_source)
        self.assertIn("grid-template-columns: 1fr", theme_source)
        self.assertNotIn('label input[type="radio"]', theme_source)
        self.assertNotIn("repeat(auto-fit", theme_source)
        self.assertIn(".workbench-page-router > .tab-wrapper", theme_source)
        self.assertIn("display: none !important", theme_source)
        self.assertNotIn("workbench-index-summary", theme_source)
        self.assertNotIn("workbench-launcher-summary", theme_source)
        self.assertNotIn("workbench-tool-launcher", theme_source)
        self.assertNotIn("render_page_switcher_group_styles", theme_source)
        self.assertNotIn("render_lab_group_rail", theme_source)
        self.assertNotIn("grid-template-columns: 280px minmax(0, 1fr)", theme_source)
        self.assertNotIn("max-height: max(420px, calc(100vh - 360px))", theme_source)
        self.assertNotIn(".workbench-page-switcher > .tab-wrapper {\n    background", theme_source)

    def test_project_has_agent_readable_design_md(self):
        design_path = ROOT / "DESIGN.md"
        self.assertTrue(design_path.exists(), "DESIGN.md should give agents persistent UI context")

        design_source = design_path.read_text(encoding="utf-8")
        required_terms = [
            "LLM Tools Workbench",
            "VoltAgent/awesome-design-md",
            "Vercel",
            "Linear",
            "Raycast",
            "command surface",
            "Long scrolling sidebars",
            "Horizontal scrolling tab strips",
            "DESIGN.md",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, design_source)

    def test_page_headers_are_strips_not_cards(self):
        theme_source = (ROOT / "workbench_theme.py").read_text(encoding="utf-8")

        self.assertIn("border-bottom: 1px solid var(--od-border)", theme_source)
        self.assertIn("border-radius: 0 !important", theme_source)
        self.assertIn("background: transparent !important", theme_source)
        self.assertNotIn(".workbench-page-hero,\n.main-header {\n    background: var(--od-bg) !important;\n    border: 1px solid var(--od-border-soft) !important;", theme_source)

    def test_app_does_not_unconditionally_import_missing_inference_lab(self):
        app_source = (ROOT / "app_gradio.py").read_text(encoding="utf-8")

        self.assertNotIn("from inference_lab import", app_source)
        self.assertIn("optional_lab_available", app_source)

    def test_page_initializers_are_lazy_loaded_by_tab(self):
        app_source = (ROOT / "app_gradio.py").read_text(encoding="utf-8")

        self.assertIn("bind_lazy_load_event", app_source)
        self.assertIn(".select(", app_source)
        self.assertNotIn("combined_load", app_source)
        self.assertNotIn("app.load(fn=", app_source)

    def test_plot_selection_is_gradio_6_compatible(self):
        trace_source = (ROOT / "agent_trace_lab" / "trace_viewer.py").read_text(encoding="utf-8")

        self.assertIn('hasattr(timeline_plot, "select")', trace_source)

    def test_eval_benchmark_plots_have_initial_values(self):
        benchmark_source = (ROOT / "eval_lab" / "benchmark_explorer.py").read_text(encoding="utf-8")

        self.assertIn("gr.Plot(value=initial_radar", benchmark_source)
        self.assertIn("gr.Plot(value=initial_bar", benchmark_source)
        self.assertIn("gr.Plot(value=initial_heatmap", benchmark_source)
        self.assertNotIn("radar_plot.value = initial_radar", benchmark_source)

    def test_key_pages_use_shared_tool_shell_pattern(self):
        page_paths = [
            ROOT / "token_lab" / "playground.py",
            ROOT / "generation_lab" / "logits_inspector.py",
            ROOT / "eval_lab" / "benchmark_explorer.py",
            ROOT / "agent_trace_lab" / "trace_viewer.py",
        ]

        for page_path in page_paths:
            source = page_path.read_text(encoding="utf-8")
            with self.subTest(page=page_path.name):
                self.assertIn("workbench-tool-shell", source)
                self.assertIn("workbench-control-panel", source)
                self.assertIn("workbench-output-panel", source)

    def test_representative_legacy_pages_use_workbench_layout(self):
        page_paths = [
            ROOT / "embedding_lab" / "model_comparison.py",
            ROOT / "data_lab" / "hf_dataset_viewer.py",
            ROOT / "model_lab" / "memory_estimator.py",
            ROOT / "finetune_lab" / "training_cost_estimator.py",
        ]

        for page_path in page_paths:
            source = page_path.read_text(encoding="utf-8")
            with self.subTest(page=page_path.name):
                self.assertIn("workbench-page-hero", source)
                self.assertIn("workbench-tool-shell", source)
                self.assertIn("workbench-control-panel", source)
                self.assertIn("workbench-output-panel", source)
                self.assertNotIn('gr.Markdown("# ', source)

    def test_unmigrated_pages_are_wrapped_by_app_fallback_shell(self):
        app_source = (ROOT / "app_gradio.py").read_text(encoding="utf-8")
        theme_source = (ROOT / "workbench_theme.py").read_text(encoding="utf-8")

        self.assertIn("_page_uses_workbench_layout", app_source)
        self.assertIn("_render_page_with_frame", app_source)
        self.assertIn("render_legacy_page_header", app_source)
        self.assertIn("render_legacy_page_context", app_source)
        self.assertIn("workbench-legacy-page-shell", theme_source)
        self.assertIn("workbench-legacy-page-context", theme_source)
        self.assertIn("workbench-legacy-output-panel", theme_source)

    def test_known_unmigrated_pages_are_detected_as_legacy(self):
        from app_gradio import _page_uses_workbench_layout

        legacy_pages = [
            {"module": "token_lab.arena"},
            {"module": "generation_lab.beam_visualizer"},
            {"module": "interpretability_lab.rope_explorer"},
        ]
        for page in legacy_pages:
            with self.subTest(module=page["module"]):
                self.assertFalse(_page_uses_workbench_layout(page))

    def test_model_ops_pages_use_native_workbench_layout(self):
        page_paths = [
            ROOT / "model_lab" / "peft_calculator.py",
            ROOT / "model_lab" / "config_diff.py",
            ROOT / "finetune_lab" / "lora_explorer.py",
        ]

        for page_path in page_paths:
            source = page_path.read_text(encoding="utf-8")
            with self.subTest(page=page_path.name):
                self.assertIn("workbench-page-hero", source)
                self.assertIn("workbench-tool-shell", source)
                self.assertIn("workbench-control-panel", source)
                self.assertIn("workbench-output-panel", source)
                self.assertNotIn('gr.Markdown("# ', source)

    def test_peft_calculator_uses_english_interaction_states(self):
        source = (ROOT / "model_lab" / "peft_calculator.py").read_text(encoding="utf-8")

        self.assertIn('input_mode == "Preset Model"', source)
        self.assertIn('return "Configuration loaded"', source)
        self.assertIn('"Please select at least one target module"', source)
        self.assertNotIn('"预设模型"', source)
        self.assertNotIn("Please select至少", source)


if __name__ == "__main__":
    unittest.main()
