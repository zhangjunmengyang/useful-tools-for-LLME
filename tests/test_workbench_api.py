"""Stateless HTTP API tests for Mechanics Explorer."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from workbench_api.app import app


class WorkbenchApiTest(unittest.TestCase):
    """验证 FastAPI 服务暴露普通 JSON 工具接口。"""

    def setUp(self):
        self.client = TestClient(app)

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

    def test_export_endpoint_writes_artifact_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_response = self.client.post(
                "/api/tools/eval_metrics/run",
                json={
                    "predictions": ["Paris"],
                    "references": ["Paris"],
                    "output_dir": tmp,
                },
            )

            self.assertEqual(run_response.status_code, 200)
            run_payload = run_response.json()
            self.assertEqual(run_payload["status"], "success")
            self.assertNotIn("artifact", run_payload)
            self.assertEqual(list(Path(tmp).iterdir()), [])

            response = self.client.post(
                "/api/tools/eval_metrics/export",
                json={
                    "inputs": {"predictions": ["Paris"], "references": ["Paris"]},
                    "output_dir": tmp,
                },
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "success")
            self.assertIn("artifact", payload)
            self.assertTrue(Path(payload["artifact"]["json_path"]).is_file())
            self.assertTrue(Path(payload["artifact"]["markdown_path"]).is_file())

    def test_unknown_tool_returns_404(self):
        response = self.client.post("/api/tools/not_real/run", json={})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Unknown tool: not_real")


if __name__ == "__main__":
    unittest.main()
