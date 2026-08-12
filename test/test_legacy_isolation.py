from pathlib import Path
import unittest


SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
LEGACY_DIRECTORY = SOURCE_DIRECTORY / "legacy"


class LegacyIsolationTests(unittest.TestCase):
    def test_legacy_package_contains_no_hdf_writer(self):
        self.assertFalse((LEGACY_DIRECTORY / "io" / "hdf_writer.py").exists())
        process_source = (
            LEGACY_DIRECTORY / "process_granule_old.py"
        ).read_text()
        self.assertNotIn("write_hdf", process_source)
        self.assertNotIn("SDSData", process_source)


if __name__ == "__main__":
    unittest.main()
