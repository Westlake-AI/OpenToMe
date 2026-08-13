import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CVScaffoldingCleanupTest(unittest.TestCase):
    def test_custom_deit_wrapper_is_not_imported(self):
        trainer_paths = (
            ROOT / "trainer/classification/in1k_trainer.py",
            ROOT / "trainer/classification/Add_module_ablation/in1k_trainer.py",
        )

        for path in trainer_paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_modules = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            }
            imported_modules.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            self.assertFalse(
                any(module.startswith("opentome.models.deit") for module in imported_modules),
                path,
            )

    def test_removed_cv_scaffolding_stays_removed(self):
        removed_paths = (
            ROOT / "opentome/models/deit/__init__.py",
            ROOT / "opentome/models/deit/deit.py",
            ROOT / "trainer/classification/c100_trainer_deit.sh",
        )
        for path in removed_paths:
            self.assertFalse(path.exists(), path)

        model_path = ROOT / "opentome/models/mergenet/model.py"
        tree = ast.parse(model_path.read_text(encoding="utf-8"), filename=str(model_path))
        script_entrypoints = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ]
        self.assertEqual(script_entrypoints, [])


if __name__ == "__main__":
    unittest.main()
