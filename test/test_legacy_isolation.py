import ast
from pathlib import Path
import unittest


SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
LEGACY_DIRECTORY = SOURCE_DIRECTORY / "legacy"


class LegacyIsolationTests(unittest.TestCase):
    def test_legacy_package_does_not_import_refactored_package(self):
        offenders = []

        for path in LEGACY_DIRECTORY.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue

                if any(module.startswith("twod_mcda") for module in modules):
                    offenders.append(path.relative_to(SOURCE_DIRECTORY))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
