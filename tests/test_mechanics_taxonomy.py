"""Mechanics Explorer taxonomy tests."""

from __future__ import annotations

import unittest

from workbench_tools.mechanics import (
    MECHANICS_CATEGORIES,
    VALID_MECHANICS_CATEGORY_IDS,
    enrich_tool_spec,
)
from workbench_tools.registry import get_registry


class MechanicsTaxonomyTest(unittest.TestCase):
    """确保工具注册表能映射到 Mechanics Explorer 的一级导航。"""

    def test_pipeline_rail_has_approved_categories(self):
        category_ids = [category["id"] for category in MECHANICS_CATEGORIES]

        self.assertEqual(
            category_ids,
            [
                "input_tokens",
                "representation_space",
                "probability_decoding",
                "transformer_anatomy",
                "data_context",
                "adaptation_cost",
                "evaluation_traces",
            ],
        )

    def test_every_registered_tool_has_valid_mechanics_metadata(self):
        for spec in get_registry().list_specs():
            with self.subTest(tool_id=spec.id):
                self.assertIsNotNone(spec.mechanics_category)
                self.assertIn(spec.mechanics_category, VALID_MECHANICS_CATEGORY_IDS)
                self.assertIsInstance(spec.mechanics_stage, int)
                self.assertGreaterEqual(spec.mechanics_stage, 1)

    def test_every_pipeline_category_has_a_registered_tool(self):
        category_ids = {category["id"] for category in MECHANICS_CATEGORIES}
        counts_by_category = {category_id: 0 for category_id in category_ids}
        for spec in get_registry().list_specs():
            counts_by_category[spec.mechanics_category] += 1

        self.assertEqual(
            {category_id for category_id, count in counts_by_category.items() if count == 0},
            set(),
        )

    def test_enriched_tool_spec_includes_category_details(self):
        spec = get_registry().get_spec("dataset_quality_check")
        payload = enrich_tool_spec(spec)

        self.assertEqual(payload["mechanics_category"], "data_context")
        self.assertEqual(payload["mechanics_category_label"], "Data & Context")
        self.assertIn("input_schema", payload)
        self.assertIn("output_schema", payload)


if __name__ == "__main__":
    unittest.main()
