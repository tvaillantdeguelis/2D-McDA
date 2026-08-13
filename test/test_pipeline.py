from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from twod_mcda.caliop.discovery import get_caliop_folder
from twod_mcda.pipeline import (
    _print_request,
    resolve_processing_request,
    run_granule_pipeline,
)
from twod_mcda.utils.timing import timer


CURRENT = Path(
    "/data/2010_01_01/"
    "CAL_LID_L1-Standard-V5-00.2010-01-01T00-22-28ZN.hdf"
)
PREVIOUS = Path(
    "/data/2009_12_31/"
    "CAL_LID_L1-Standard-V5-00.2009-12-31T23-29-53ZD.hdf"
)
NEXT = Path(
    "/data/2010_01_01/"
    "CAL_LID_L1-Standard-V5-00.2010-01-01T01-15-03ZN.hdf"
)


def config():
    return {
        "granule": "2010-01-01T00-22-28ZN",
        "cal_lid_l1": {
            "root_directory": "/data",
            "version": "5.00",
            "path_format": (
                "CAL_LID_L1.v{version}/{year}/"
                "{year}_{month:02d}_{day:02d}"
            ),
        },
        "processing": {
            "save_development_data": False,
            "max_altitude_km": 30,
        },
        "output": {
            "root_directory": "/output",
            "product_type": "Dev",
            "path_format": (
                "2D_McDA.v{version}/{year}/"
                "{year}_{month:02d}_{day:02d}"
            ),
        },
        "subset": {
            "mode": "profindex",
            "start": None,
            "end": None,
        },
    }


class PipelineTests(unittest.TestCase):
    def test_timer_uses_four_spaces_for_nested_levels(self):
        output = StringIO()

        with redirect_stdout(output):
            with timer("outer"):
                with timer("inner"):
                    pass

        self.assertIn("outer...\n    inner...", output.getvalue())
        self.assertNotIn("\tinner", output.getvalue())

    def test_print_request_displays_resolved_profile_limits(self):
        request = SimpleNamespace(
            output_version="V2.0.0",
            caliop_version="V5.00",
            save_development_data=False,
            maximum_altitude_km=30,
            subset_active=True,
            subset_mode="profindex",
            subset_start=0,
            subset_end=None,
            previous_granule=None,
            next_granule=None,
        )
        granule = SimpleNamespace(
            filepath="/data/current.hdf",
            prof_min=0,
            prof_max=9999,
        )

        output = StringIO()
        with redirect_stdout(output):
            _print_request(request, granule, None, None, 10000, 2)

        self.assertIn("Profile limits         : 0 -> 9999", output.getvalue())

        request.subset_active = False
        output = StringIO()
        with redirect_stdout(output):
            _print_request(request, granule, None, None, 10000, 2)

        self.assertIn("Subset mode            : false", output.getvalue())
        self.assertNotIn("limits", output.getvalue())

    def test_caliop_folder_uses_configured_root_and_path_format(self):
        folder = get_caliop_folder(
            config(),
            datetime(2010, 1, 1),
        )

        self.assertEqual(
            folder,
            Path("/data/CAL_LID_L1.v5.00/2010/2010_01_01"),
        )

    @patch("twod_mcda.pipeline.find_neighbor_granules", return_value=(PREVIOUS, NEXT))
    @patch("twod_mcda.pipeline.find_granule_file", return_value=CURRENT)
    @patch("twod_mcda.pipeline.get_full_version", return_value="v2.0.0")
    def test_processing_request_maps_paths_and_versions(
        self,
        get_full_version,
        find_granule_file,
        find_neighbor_granules,
    ):
        request = resolve_processing_request(config())

        self.assertEqual(
            request.granule_date,
            "2010-01-01T00-22-28ZN",
        )
        self.assertEqual(request.caliop_version, "V5.00")
        self.assertEqual(request.current_directory, Path("/data/2010_01_01"))
        self.assertEqual(
            request.previous_granule,
            "2009-12-31T23-29-53ZD",
        )
        self.assertEqual(
            request.previous_directory,
            Path("/data/2009_12_31"),
        )
        self.assertEqual(
            request.next_granule,
            "2010-01-01T01-15-03ZN",
        )
        self.assertEqual(request.output_version, "V2.0.0")
        self.assertEqual(request.output_product_type, "Dev")
        self.assertTrue(request.subset_active)
        self.assertEqual(request.subset_mode, "profindex")
        self.assertEqual(request.maximum_altitude_km, 30)
        self.assertEqual(request.maximum_altitude_index, 600)
        self.assertEqual(
            request.output_directory,
            Path("/output/2D_McDA.v2.0.0/2010/2010_01_01"),
        )

    @patch("twod_mcda.pipeline.find_neighbor_granules", return_value=(PREVIOUS, NEXT))
    @patch("twod_mcda.pipeline.find_granule_file", return_value=CURRENT)
    @patch("twod_mcda.pipeline.get_full_version", return_value="v2.0.0")
    def test_processing_request_disables_absent_or_inactive_subset(
        self,
        get_full_version,
        find_granule_file,
        find_neighbor_granules,
    ):
        configurations = []

        without_subset = config()
        without_subset.pop("subset")
        configurations.append(without_subset)

        inactive_subset = config()
        inactive_subset["subset"] = {
            "activate": False,
            "mode": "longitude",
            "start": 63.2,
            "end": 61.3,
        }
        configurations.append(inactive_subset)

        for cfg in configurations:
            with self.subTest(subset=cfg.get("subset")):
                request = resolve_processing_request(cfg)

                self.assertFalse(request.subset_active)
                self.assertEqual(request.subset_mode, "profindex")
                self.assertIsNone(request.subset_start)
                self.assertIsNone(request.subset_end)

    @patch("twod_mcda.pipeline.find_neighbor_granules", return_value=(PREVIOUS, NEXT))
    @patch("twod_mcda.pipeline.find_granule_file", return_value=CURRENT)
    @patch("twod_mcda.pipeline.get_full_version", return_value="v2.0.0")
    def test_processing_request_supports_40_km(
        self,
        get_full_version,
        find_granule_file,
        find_neighbor_granules,
    ):
        cfg = config()
        cfg["processing"]["max_altitude_km"] = 40

        request = resolve_processing_request(cfg)

        self.assertIsNone(request.maximum_altitude_index)

    @patch("twod_mcda.pipeline.find_neighbor_granules", return_value=(PREVIOUS, NEXT))
    @patch("twod_mcda.pipeline.find_granule_file", return_value=CURRENT)
    def test_processing_request_rejects_unsupported_altitude(
        self,
        find_granule_file,
        find_neighbor_granules,
    ):
        cfg = config()
        cfg["processing"]["max_altitude_km"] = 35

        with self.assertRaisesRegex(
            ValueError,
            "max_altitude_km must be either 30 or 40",
        ):
            resolve_processing_request(cfg)

    @patch("twod_mcda.pipeline._execute_pipeline", return_value=Path("/output/result.nc"))
    @patch("twod_mcda.pipeline.resolve_processing_request")
    def test_run_granule_pipeline_executes_resolved_request(
        self,
        resolve_request,
        execute_pipeline,
    ):
        resolved = object()
        resolve_request.return_value = resolved

        result = run_granule_pipeline(config())

        execute_pipeline.assert_called_once_with(resolved)
        self.assertEqual(result, Path("/output/result.nc"))

if __name__ == "__main__":
    unittest.main()
