from datetime import datetime
from pathlib import Path
import unittest
from unittest.mock import patch

from twod_mcda.caliop.discovery import get_caliop_folder
from twod_mcda.pipeline import (
    resolve_processing_request,
    run_granule_pipeline,
)


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
