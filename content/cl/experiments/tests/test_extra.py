from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from learn_cl_experiments.extra.registry import EXTRAS  # noqa: E402
from learn_cl_experiments.extra_core import validate_extra_payload
from learn_cl_experiments.gpu_recipes import RECIPES, get_recipe


class ExtraExperimentsTest(unittest.TestCase):
    def test_all_extras_pass(self) -> None:
        self.assertEqual(
            list(EXTRAS),
            [
                "unplug",
                "distill",
                "skill",
                "selfedit",
                "conflict",
                "evolve",
                "route",
                "capacity",
                "sleep",
                "surprise",
                "seqedit",
                "onpolicy",
                "ortho",
                "ewcmem",
                "plastic",
                "graduate",
                "buffer",
                "gendream",
                "stale",
                "shadow",
                "eligible",
                "budget",
                "tombstone",
                "longtail",
                "compose",
                "disagree",
                "keepfail",
                "rollback",
            ],
        )
        for extra in EXTRAS.values():
            payload = extra.execute()
            validate_extra_payload(payload, extra)
            failed = [name for name, ok in payload["checks"].items() if not ok]
            self.assertEqual(failed, [], extra.extra_id)


class GpuRecipesTest(unittest.TestCase):
    def test_recipe_ids_unique(self) -> None:
        ids = [recipe.recipe_id for recipe in RECIPES]
        self.assertEqual(len(ids), len(set(ids)))
        get_recipe("seal")
        get_recipe("razor-mnist")
        get_recipe("hipporag")
        get_recipe("memoryllm")
        get_recipe("voyager")
        get_recipe("vandeven-dgr")
        get_recipe("inflora")
        get_recipe("easyedit-wise")
        get_recipe("unsloth-lora")
        get_recipe("langmem")


if __name__ == "__main__":
    unittest.main()
