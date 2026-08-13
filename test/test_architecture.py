import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "twod_mcda"


class PackageArchitectureTests(unittest.TestCase):
    def test_expected_layers_exist(self):
        for package in ("algorithm", "caliop", "workflow", "output", "utils"):
            self.assertTrue((PACKAGE_ROOT / package / "__init__.py").is_file())

    def test_caliop_altitude_resource_is_colocated(self):
        resource = (
            PACKAGE_ROOT
            / "caliop"
            / "resources"
            / "lidar_data_altitudes.pkl"
        )
        self.assertTrue(resource.is_file())
        self.assertFalse((PACKAGE_ROOT / "lidar_data_altitudes.pkl").exists())

        packaging = (PROJECT_ROOT / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '"caliop/resources/lidar_data_altitudes.pkl"',
            packaging,
        )

    def test_removed_legacy_modules_do_not_return(self):
        for path in (
            "calipso_calculator.py",
            "calipso_constants.py",
            "config.py",
            "workflow/processor.py",
        ):
            self.assertFalse((PACKAGE_ROOT / path).exists(), path)

        for package in (
            "detection",
            "io",
            "merge",
            "models",
            "preprocessing",
            "processing",
        ):
            self.assertEqual(list((PACKAGE_ROOT / package).glob("*.py")), [])

    def test_package_does_not_use_star_imports(self):
        offenders = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if any(alias.name == "*" for alias in node.names):
                        offenders.append(path.relative_to(PROJECT_ROOT))
        self.assertEqual(offenders, [])

    def test_lower_layers_do_not_import_workflow_or_output(self):
        forbidden = {
            "algorithm": {"workflow", "output"},
            "caliop": {"algorithm", "workflow", "output"},
            "output": {"algorithm", "workflow"},
            "utils": {"algorithm", "caliop", "workflow", "output"},
        }
        offenders = []

        for layer, forbidden_layers in forbidden.items():
            for path in (PACKAGE_ROOT / layer).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom) or not node.module:
                        continue
                    parts = node.module.split(".")
                    if parts[:1] != ["twod_mcda"] or len(parts) < 2:
                        continue
                    if parts[1] in forbidden_layers:
                        offenders.append(
                            (str(path.relative_to(PROJECT_ROOT)), node.module)
                        )

        self.assertEqual(offenders, [])

    def test_xarray_is_deferred(self):
        environment = (PROJECT_ROOT / "environment.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("xarray", environment)
        for path in PACKAGE_ROOT.rglob("*.py"):
            self.assertNotIn("import xarray", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
