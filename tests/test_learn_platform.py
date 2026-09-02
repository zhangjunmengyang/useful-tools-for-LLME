"""Multi-topic learn catalog: four switcher entries, readable lessons."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from learn_platform.catalog import list_topics, topic_lesson, topic_outline
from workbench_api.app import app


class LearnPlatformTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_switcher_has_exactly_four_topics_in_order(self):
        topics = list_topics()
        self.assertEqual([topic["id"] for topic in topics], ["omni", "wm", "cl", "llm"])
        self.assertEqual([topic["title"] for topic in topics], ["Omni", "世界模型", "持续学习", "LLM"])

    def test_api_topics_match_switcher(self):
        response = self.client.get("/api/learn/topics")
        self.assertEqual(response.status_code, 200)
        ids = [topic["id"] for topic in response.json()["topics"]]
        self.assertEqual(ids, ["omni", "wm", "cl", "llm"])

    def test_existing_topics_have_readable_lessons(self):
        for topic_id in ("omni", "wm", "cl"):
            outline = topic_outline(topic_id)
            self.assertTrue(outline["ready"], msg=f"{topic_id} not ready: {outline}")
            self.assertGreaterEqual(len(outline["lessons"]), 8, msg=topic_id)
            self.assertNotIn("other", [unit["id"] for unit in outline["units"]], msg=topic_id)
            lesson = topic_lesson(topic_id, outline["default_lesson_id"])
            self.assertGreater(len(lesson["read"]), 200, msg=topic_id)
            self.assertGreater(len(lesson["learn"]), 40, msg=topic_id)

    def test_llm_lessons_include_play_tools(self):
        outline = topic_outline("llm")
        self.assertEqual(outline["default_lesson_id"], "tokens")
        self.assertEqual(len(outline["lessons"]), 7)
        lesson = topic_lesson("llm", "tokens")
        self.assertIn("unicode_analyze", lesson["play_tools"])
        self.assertIn("tokenizer_encode", lesson["play_tools"])
        rope = topic_lesson("llm", "rope")
        self.assertIn("rope_frequencies", rope["play_tools"])

    def test_llm_lessons_have_english_bodies(self):
        outline = topic_outline("llm")
        self.assertEqual(outline["title_en"], "LLM")
        for lesson_id in ("tokens", "rope", "lora"):
            lesson = topic_lesson("llm", lesson_id)
            self.assertEqual(lesson["body_locale"], "both", msg=lesson_id)
            self.assertTrue(lesson["read_en"], msg=lesson_id)
            self.assertTrue(lesson["title_en"], msg=lesson_id)

    def test_imported_lessons_stay_chinese_only(self):
        for topic_id in ("omni", "wm", "cl"):
            outline = topic_outline(topic_id)
            lesson = topic_lesson(topic_id, outline["default_lesson_id"])
            self.assertEqual(lesson["body_locale"], "zh", msg=topic_id)
            self.assertIsNone(lesson["read_en"])
            self.assertGreater(len(lesson["read"]), 200, msg=topic_id)

    def test_unknown_topic_is_404(self):
        response = self.client.get("/api/learn/topics/quant")
        self.assertEqual(response.status_code, 404)

    def test_labs_catalog_keeps_every_gradio_page(self):
        response = self.client.get("/api/labs")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        ids = {page["id"] for page in payload["pages"]}
        for page_id in (
            "toolbox_tool_runner",
            "token_playground",
            "token_arena",
            "token_chat_template",
            "embedding_vector_arithmetic",
            "embedding_model_comparison",
            "embedding_visualization",
            "embedding_semantic_similarity",
            "generation_logits",
            "generation_beam",
            "generation_kv_cache",
            "interpretability_attention",
            "interpretability_rope",
            "interpretability_ffn",
            "data_dataset_viewer",
            "data_cleaner",
            "data_formatter",
            "rag_chunking",
            "rag_retrieval",
            "model_memory",
            "model_peft",
            "model_config_diff",
            "finetune_lora",
            "finetune_training_cost",
            "agent_trace_viewer",
            "agent_trace_analyzer",
            "eval_benchmark",
            "eval_llm_judge",
            "eval_pipeline",
        ):
            self.assertIn(page_id, ids, msg=page_id)
        arena = next(page for page in payload["pages"] if page["id"] == "token_arena")
        self.assertEqual(arena["embed_url"], "/labs/?lab=token_arena")
        tools = self.client.get("/api/tools")
        self.assertEqual(tools.status_code, 200)
        self.assertEqual(len(tools.json()["tools"]), 17)


if __name__ == "__main__":
    unittest.main()
