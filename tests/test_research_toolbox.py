"""Research toolbox public interface tests."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ResearchToolboxTest(unittest.TestCase):
    """验证研究工具箱接口可被 UI、CLI 和外部 Agent 复用。"""

    def test_public_schemas_are_json_compatible(self):
        from workbench_tools import LabCapability, ToolArtifact, ToolRun, ToolSpec

        spec = ToolSpec(
            id="eval_metrics",
            label="Evaluation Metrics",
            description="Compute common text-generation evaluation metrics.",
            lab="Eval Lab",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            concepts=["evaluation", "metrics"],
            requires_model_download=False,
        )
        artifact = ToolArtifact(
            markdown_path="/tmp/report.md",
            json_path="/tmp/report.json",
        )
        run = ToolRun(
            tool_id="eval_metrics",
            status="success",
            inputs={"predictions": ["Paris"], "references": ["Paris"]},
            result={"exact_match": 1.0},
            duration_ms=1.5,
            artifact=artifact,
        )
        capability = LabCapability(
            page_id="eval_pipeline",
            page_label="Eval Pipeline",
            lab="Eval Lab",
            tool_ids=["eval_metrics"],
            concepts=["evaluation"],
        )

        payload = {
            "spec": spec.to_dict(),
            "run": run.to_dict(),
            "capability": capability.to_dict(),
        }

        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertIn("eval_metrics", encoded)
        self.assertEqual(payload["run"]["artifact"]["json_path"], "/tmp/report.json")

    def test_registry_runs_eval_metrics_as_tool_run(self):
        from workbench_tools.registry import get_registry

        run = get_registry().run(
            "eval_metrics",
            {"predictions": ["Paris"], "references": ["Paris"]},
            export=False,
        )

        self.assertEqual(run.status, "success")
        self.assertIsNone(run.error)
        self.assertEqual(run.result["exact_match"], 1.0)
        self.assertEqual(run.result["f1"], 1.0)
        self.assertGreaterEqual(run.duration_ms, 0)

    def test_registry_validates_inputs_before_running_tools(self):
        from workbench_tools.registry import get_registry

        run = get_registry().run(
            "eval_metrics",
            {"predictions": ["Paris"]},
            export=False,
        )

        self.assertEqual(run.status, "error")
        self.assertIn("references", run.error)
        self.assertEqual(run.result, {})

    def test_registry_exposes_expanded_toolbox(self):
        from workbench_tools.registry import get_registry

        tool_ids = {spec.id for spec in get_registry().list_specs()}

        expected = {
            "tokenizer_encode",
            "unicode_analyze",
            "sampling_distribution",
            "kv_cache_growth",
            "rope_frequencies",
            "ffn_activation_compare",
            "data_clean",
            "dataset_quality_check",
            "instruct_format",
            "rag_lexical_retrieval",
            "lora_params_estimate",
            "training_cost_estimate",
            "eval_metrics",
            "kv_cache_estimate",
            "rag_chunk",
            "trace_analyze",
        }
        self.assertTrue(expected.issubset(tool_ids))
        self.assertGreaterEqual(len(tool_ids), 16)

    def test_non_model_tools_are_callable_without_gradio(self):
        from workbench_tools.registry import get_registry

        sampling = get_registry().run(
            "sampling_distribution",
            {"logits": [2.0, 1.0, 0.0], "tokens": ["A", "B", "C"], "top_k": 2},
        )
        cleaning = get_registry().run(
            "data_clean",
            {"text": "<p>Hello</p> https://example.com", "rules": ["html", "url"]},
        )
        lora = get_registry().run(
            "lora_params_estimate",
            {
                "hidden_size": 128,
                "num_layers": 2,
                "num_heads": 4,
                "num_kv_heads": 4,
                "intermediate_size": 256,
                "rank": 8,
                "target_modules": ["q_proj", "v_proj"],
            },
        )
        cost = get_registry().run(
            "training_cost_estimate",
            {
                "model_params": 1000000,
                "tokens": 1000,
                "gpu_tflops": 100,
                "cost_per_hour": 2,
            },
        )
        unicode_run = get_registry().run("unicode_analyze", {"text": "Ａ café"})
        rope = get_registry().run("rope_frequencies", {"dim": 8, "max_position": 4})
        ffn = get_registry().run(
            "ffn_activation_compare",
            {"x_values": [-1.0, 0.0, 1.0]},
        )
        retrieval = get_registry().run(
            "rag_lexical_retrieval",
            {
                "query": "python language",
                "documents": ["Python is a programming language.", "Coffee is a drink."],
                "top_k": 1,
            },
        )
        quality = get_registry().run(
            "dataset_quality_check",
            {
                "samples": [
                    {"instruction": "Say hi", "output": "Hi"},
                    {"instruction": "", "output": "Missing instruction"},
                    {"instruction": "Say hi", "output": "Hi"},
                ],
                "text_fields": ["instruction", "output"],
            },
        )

        self.assertEqual(sampling.status, "success")
        self.assertEqual(len(sampling.result["distribution"]), 3)
        self.assertEqual(cleaning.result["cleaned_text"], "Hello")
        self.assertEqual(lora.result["total_params"], 8192)
        self.assertEqual(cost.result["total_flops"], 6000000000)
        self.assertFalse(unicode_run.result["normalization"]["nfkc_equal"])
        self.assertEqual(rope.result["freq_shape"], [4, 4])
        self.assertIn("GELU", ffn.result["activations"])
        self.assertEqual(retrieval.result["results"][0]["document_index"], 0)
        self.assertEqual(quality.result["duplicate_count"], 1)
        json.dumps(sampling.to_dict(), allow_nan=False)

    def test_artifact_export_writes_markdown_and_json_inside_output_dir(self):
        from workbench_tools.registry import get_registry

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run = get_registry().run(
                "eval_metrics",
                {"predictions": ["Paris"], "references": ["Paris"]},
                export=True,
                output_dir=output_dir,
            )

            self.assertEqual(run.status, "success")
            self.assertIsNotNone(run.artifact)
            markdown_path = Path(run.artifact.markdown_path)
            json_path = Path(run.artifact.json_path)
            self.assertTrue(markdown_path.is_file())
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.resolve().is_relative_to(output_dir.resolve()))
            self.assertTrue(json_path.resolve().is_relative_to(output_dir.resolve()))
            self.assertIn("Evaluation Metrics", markdown_path.read_text(encoding="utf-8"))
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["result"]["exact_match"], 1.0)
            self.assertIn("summary", saved)
            self.assertIn("key_findings", saved)
            self.assertIn("limitations", saved)
            self.assertIn("reproduce_command", saved)
            self.assertEqual(saved["source_page"], "eval_pipeline")
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("## Key Findings", markdown)
            self.assertIn("## Reproduce", markdown)

    def test_cli_lists_and_runs_core_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.json"
            config_path.write_text(
                json.dumps(
                    {"predictions": ["Paris"], "references": ["Paris"]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            list_result = subprocess.run(
                [sys.executable, "-m", "workbench_tools", "list"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("eval_metrics", list_result.stdout)

            run_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "workbench_tools",
                    "run",
                    "eval_metrics",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(tmp_path / "research"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(run_result.stdout)
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["result"]["exact_match"], 1.0)
            self.assertTrue(Path(payload["artifact"]["json_path"]).is_file())

    def test_cli_inspect_and_batch_run_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            batch_path = tmp_path / "batch.jsonl"
            batch_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "tool_id": "eval_metrics",
                                "inputs": {
                                    "predictions": ["Paris"],
                                    "references": ["Paris"],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "tool_id": "kv_cache_estimate",
                                "inputs": {
                                    "num_layers": 2,
                                    "hidden_size": 128,
                                    "num_heads": 4,
                                    "seq_length": 16,
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            inspect_result = subprocess.run(
                [sys.executable, "-m", "workbench_tools", "inspect", "eval_metrics"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"input_schema"', inspect_result.stdout)

            batch_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "workbench_tools",
                    "batch",
                    str(batch_path),
                    "--no-export",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(batch_result.stdout)
            self.assertEqual(len(payload["runs"]), 2)
            self.assertEqual(payload["runs"][0]["status"], "success")

    def test_cli_manifest_and_sample_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.json"

            manifest_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "workbench_tools",
                    "manifest",
                    "--output",
                    str(manifest_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(manifest_result.stdout)
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(payload["tools"]), 16)
            self.assertGreaterEqual(len(payload["capabilities"]), 11)
            self.assertEqual(payload["tools"][0]["id"], saved["tools"][0]["id"])

            sample_result = subprocess.run(
                [sys.executable, "-m", "workbench_tools", "sample-config", "rag_lexical_retrieval"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            sample = json.loads(sample_result.stdout)
            self.assertIn("query", sample)
            self.assertIn("documents", sample)

    def test_default_configs_validate_for_all_tools(self):
        from toolbox_lab.tool_runner import DEFAULT_CONFIGS
        from workbench_tools.registry import get_registry
        from workbench_tools.validation import validate_against_schema

        registry = get_registry()
        for spec in registry.list_specs():
            with self.subTest(tool_id=spec.id):
                self.assertIn(spec.id, DEFAULT_CONFIGS)
                errors = validate_against_schema(DEFAULT_CONFIGS[spec.id], spec.input_schema)
                self.assertEqual(errors, [])

    def test_lab_capabilities_map_pages_to_tools(self):
        from workbench_tools.capabilities import get_lab_capabilities

        capabilities = get_lab_capabilities()
        by_page = {cap.page_id: cap for cap in capabilities}

        self.assertIn("toolbox_tool_runner", by_page)
        self.assertIn("eval_metrics", by_page["eval_pipeline"].tool_ids)
        self.assertIn("kv_cache_estimate", by_page["generation_kv_cache"].tool_ids)
        self.assertIn("trace_analyze", by_page["agent_trace_analyzer"].tool_ids)
        self.assertIn("rag_lexical_retrieval", by_page["rag_retrieval"].tool_ids)
        self.assertIn("unicode_analyze", by_page["token_playground"].tool_ids)

    def test_app_registers_toolbox_page_without_agent_runtime(self):
        app_source = (ROOT / "app_gradio.py").read_text(encoding="utf-8")

        self.assertIn("toolbox_tool_runner", app_source)
        self.assertIn("toolbox_lab.tool_runner", app_source)
        self.assertNotIn("hermes_client", app_source)
        self.assertNotIn("Autoresearch Hub", app_source)

    def test_tool_runner_surfaces_schema_filters_and_cli_command(self):
        source = (ROOT / "toolbox_lab" / "tool_runner.py").read_text(encoding="utf-8")

        self.assertIn("Concept Filter", source)
        self.assertIn("Input Schema", source)
        self.assertIn("CLI Command", source)
        self.assertIn("Lab Filter", source)
        self.assertIn("Tool Catalog", source)


if __name__ == "__main__":
    unittest.main()
