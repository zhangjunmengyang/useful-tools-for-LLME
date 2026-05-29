"""Stateless HTTP API tests for Mechanics Explorer."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from workbench_api.app import app
from workbench_tools.default_configs import DEFAULT_CONFIGS


class WorkbenchApiTest(unittest.TestCase):
    """验证 FastAPI 服务暴露普通 JSON 工具接口。"""

    def setUp(self):
        self.client = TestClient(app)
        self.previous_artifact_root = getattr(app.state, "artifact_root", None)
        self.previous_frontend_dist = getattr(app.state, "frontend_dist", None)

    def tearDown(self):
        if self.previous_artifact_root is None:
            if hasattr(app.state, "artifact_root"):
                delattr(app.state, "artifact_root")
        else:
            app.state.artifact_root = self.previous_artifact_root
        if self.previous_frontend_dist is None:
            if hasattr(app.state, "frontend_dist"):
                delattr(app.state, "frontend_dist")
        else:
            app.state.frontend_dist = self.previous_frontend_dist

    def test_health_endpoint(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_tools_endpoint_returns_categories_and_tools(self):
        response = self.client.get("/api/tools")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("categories", payload)
        self.assertIn("tools", payload)
        self.assertIn("data_context", {category["id"] for category in payload["categories"]})
        dataset_tool = next(tool for tool in payload["tools"] if tool["id"] == "dataset_quality_check")
        self.assertEqual(dataset_tool["mechanics_category"], "data_context")
        self.assertEqual(dataset_tool["mechanics_category_label"], "Data & Context")

    def test_tools_endpoint_includes_runnable_sample_input(self):
        response = self.client.get("/api/tools")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        unicode_tool = next(tool for tool in payload["tools"] if tool["id"] == "unicode_analyze")
        vector_tool = next(tool for tool in payload["tools"] if tool["id"] == "vector_similarity")
        self.assertEqual(unicode_tool["sample_input"], DEFAULT_CONFIGS["unicode_analyze"])
        self.assertEqual(vector_tool["sample_input"], DEFAULT_CONFIGS["vector_similarity"])

    def test_inspect_tool_endpoint_returns_single_tool_metadata(self):
        response = self.client.get("/api/tools/dataset_quality_check")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["id"], "dataset_quality_check")
        self.assertEqual(payload["mechanics_category"], "data_context")
        self.assertEqual(payload["mechanics_category_label"], "Data & Context")
        self.assertEqual(payload["sample_input"], DEFAULT_CONFIGS["dataset_quality_check"])

    def test_inspect_unknown_tool_returns_404(self):
        response = self.client.get("/api/tools/not_real")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Unknown tool: not_real")

    def test_run_endpoint_is_stateless_and_returns_result_immediately(self):
        response = self.client.post(
            "/api/tools/unicode_analyze/run",
            json={"text": "Ａ café"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["tool_id"], "unicode_analyze")
        self.assertEqual(payload["status"], "success")
        self.assertIn("result", payload)
        self.assertNotIn("run_id", payload)
        self.assertFalse(payload["result"]["normalization"]["nfkc_equal"])

    def test_run_endpoint_returns_validation_error_without_exception_page(self):
        response = self.client.post(
            "/api/tools/eval_metrics/run",
            json={"predictions": ["Paris"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "error")
        self.assertIn("references", payload["error"])

    def test_run_endpoint_returns_structured_error_for_missing_body(self):
        response = self.client.post("/api/tools/unicode_analyze/run")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["tool_id"], "unicode_analyze")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["inputs"], {})
        self.assertEqual(payload["result"], {})
        self.assertEqual(payload["error"], "Request body must be a JSON object")
        self.assertNotIn("artifact", payload)
        self.assertNotIn("run_id", payload)

    def test_run_endpoint_returns_structured_error_for_non_object_body(self):
        response = self.client.post("/api/tools/unicode_analyze/run", json=[])

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["tool_id"], "unicode_analyze")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["inputs"], {})
        self.assertEqual(payload["result"], {})
        self.assertEqual(payload["error"], "Request body must be a JSON object")
        self.assertNotIn("artifact", payload)
        self.assertNotIn("run_id", payload)

    def test_export_endpoint_returns_structured_error_for_non_object_body(self):
        response = self.client.post("/api/tools/eval_metrics/export", json=[])

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["tool_id"], "eval_metrics")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["inputs"], {})
        self.assertEqual(payload["result"], {})
        self.assertEqual(payload["error"], "Request body must be a JSON object")
        self.assertNotIn("artifact", payload)
        self.assertNotIn("run_id", payload)

    def test_export_endpoint_writes_artifact_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            app.state.artifact_root = tmp_path / "api_artifacts"
            ignored_output_dir = tmp_path / "ignored"

            run_response = self.client.post(
                "/api/tools/eval_metrics/run",
                json={
                    "predictions": ["Paris"],
                    "references": ["Paris"],
                    "output_dir": str(ignored_output_dir),
                },
            )

            self.assertEqual(run_response.status_code, 200)
            run_payload = run_response.json()
            self.assertEqual(run_payload["status"], "success")
            self.assertNotIn("artifact", run_payload)
            self.assertFalse(ignored_output_dir.exists())

            response = self.client.post(
                "/api/tools/eval_metrics/export",
                json={
                    "inputs": {"predictions": ["Paris"], "references": ["Paris"]},
                    "output_dir": str(ignored_output_dir),
                },
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "success")
            self.assertIn("artifact", payload)
            json_path = Path(payload["artifact"]["json_path"])
            markdown_path = Path(payload["artifact"]["markdown_path"])
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertTrue(json_path.resolve().is_relative_to(app.state.artifact_root.resolve()))
            self.assertTrue(markdown_path.resolve().is_relative_to(app.state.artifact_root.resolve()))
            self.assertFalse(ignored_output_dir.exists())

    def test_export_endpoint_uses_environment_artifact_root_when_state_is_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_root = Path(tmp) / "env_artifacts"
            if hasattr(app.state, "artifact_root"):
                delattr(app.state, "artifact_root")

            with patch.dict(os.environ, {"WORKBENCH_ARTIFACT_ROOT": str(env_root)}):
                response = self.client.post(
                    "/api/tools/eval_metrics/export",
                    json={
                        "inputs": {
                            "predictions": ["Paris"],
                            "references": ["Paris"],
                        }
                    },
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "success")
            json_path = Path(payload["artifact"]["json_path"])
            markdown_path = Path(payload["artifact"]["markdown_path"])
            self.assertTrue(json_path.resolve().is_relative_to(env_root.resolve()))
            self.assertTrue(markdown_path.resolve().is_relative_to(env_root.resolve()))

    def test_unknown_tool_returns_404(self):
        response = self.client.post("/api/tools/not_real/run", json={})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Unknown tool: not_real")

    def test_serves_configured_frontend_dist_and_deep_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp) / "dist"
            assets = dist / "assets"
            assets.mkdir(parents=True)
            (dist / "index.html").write_text("<div id=\"root\">Workbench</div>", encoding="utf-8")
            (assets / "app.js").write_text("console.log('workbench');", encoding="utf-8")
            app.state.frontend_dist = dist

            root_response = self.client.get("/")
            asset_response = self.client.get("/assets/app.js")
            deep_link_response = self.client.get("/workbench/deep/link")

            self.assertEqual(root_response.status_code, 200)
            self.assertIn("Workbench", root_response.text)
            self.assertEqual(asset_response.status_code, 200)
            self.assertIn("workbench", asset_response.text)
            self.assertEqual(deep_link_response.status_code, 200)
            self.assertIn("Workbench", deep_link_response.text)


if __name__ == "__main__":
    unittest.main()
