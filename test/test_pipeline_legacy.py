from datetime import datetime
from pathlib import Path
import unittest
from unittest.mock import patch

from twod_mcda.io.granule_finder import get_caliop_folder
from twod_mcda.pipeline import _legacy_config, process_granule


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
            "product_type": "Standard",
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
            "filetype": "HDF",
        },
        "subset": {
            "mode": "profindex",
            "start": None,
            "end": None,
        },
    }


class LegacyPipelineTests(unittest.TestCase):
    def test_caliop_folder_uses_configured_root_and_path_format(self):
        folder = get_caliop_folder(
            config(),
            datetime(2010, 1, 1),
        )

        self.assertEqual(
            folder,
            Path("/data/CAL_LID_L1.v5.00/2010/2010_01_01"),
        )

    @patch("twod_mcda.pipeline.get_full_version", return_value="v2.0.0")
    def test_legacy_config_maps_paths_and_versions(self, get_full_version):
        legacy = _legacy_config(config(), CURRENT, PREVIOUS, NEXT)

        self.assertEqual(
            legacy["granule_date"],
            "2010-01-01T00-22-28ZN",
        )
        self.assertEqual(legacy["cal_lid_l1_version"], "V5.00")
        self.assertEqual(legacy["cal_lid_l1_type"], "Standard")
        self.assertEqual(legacy["folder_path"], "/data/2010_01_01")
        self.assertEqual(
            legacy["previous_granule"],
            "2009-12-31T23-29-53ZD",
        )
        self.assertEqual(
            legacy["previous_folder_path"],
            "/data/2009_12_31",
        )
        self.assertEqual(
            legacy["next_granule"],
            "2010-01-01T01-15-03ZN",
        )
        self.assertEqual(legacy["version_2d_mcda"], "V2.0.0")
        self.assertEqual(legacy["type_2d_mcda"], "Dev")
        self.assertEqual(legacy["slice_type"], "profindex")
        self.assertEqual(legacy["index30m_alt_max"], 600)
        self.assertEqual(
            legacy["out_folder"],
            "/output/2D_McDA.v2.0.0/2010/2010_01_01",
        )

    def test_legacy_config_rejects_non_hdf_output(self):
        cfg = config()
        cfg["output"]["filetype"] = "netCDF"

        with self.assertRaisesRegex(
            ValueError,
            'output.filetype must be "HDF"',
        ):
            _legacy_config(cfg, CURRENT, PREVIOUS, NEXT)

    @patch("twod_mcda.pipeline.get_full_version", return_value="v2.0.0")
    def test_legacy_config_supports_40_km(self, get_full_version):
        cfg = config()
        cfg["processing"]["max_altitude_km"] = 40

        legacy = _legacy_config(cfg, CURRENT, PREVIOUS, NEXT)

        self.assertIsNone(legacy["index30m_alt_max"])

    def test_legacy_config_rejects_unsupported_altitude(self):
        cfg = config()
        cfg["processing"]["max_altitude_km"] = 35

        with self.assertRaisesRegex(
            ValueError,
            "max_altitude_km must be either 30 or 40",
        ):
            _legacy_config(cfg, CURRENT, PREVIOUS, NEXT)

    def test_legacy_config_rejects_old_schema(self):
        cfg = config()
        cfg["processing"]["process_up_to_40km"] = False

        with self.assertRaisesRegex(
            ValueError,
            '"process_up_to_40km" was renamed to "max_altitude_km"',
        ):
            _legacy_config(cfg, CURRENT, PREVIOUS, NEXT)

    @patch("twod_mcda.pipeline.runpy.run_module")
    @patch("twod_mcda.pipeline.get_full_version", return_value="v2.0.0")
    @patch(
        "twod_mcda.pipeline.find_neighbor_granules",
        return_value=(PREVIOUS, NEXT),
    )
    @patch("twod_mcda.pipeline.find_granule_file", return_value=CURRENT)
    def test_process_granule_invokes_legacy_module(
        self,
        find_granule_file,
        find_neighbor_granules,
        get_full_version,
        run_module,
    ):
        process_granule(config())

        run_module.assert_called_once()
        args, kwargs = run_module.call_args
        self.assertEqual(args, ("legacy.process_granule_old",))
        self.assertEqual(kwargs["run_name"], "__main__")
        self.assertEqual(
            kwargs["init_globals"]["LEGACY_CONFIG"]["granule_date"],
            "2010-01-01T00-22-28ZN",
        )

    @patch("twod_mcda.processing.granule_processor.process_granule_refactored")
    @patch("twod_mcda.pipeline.get_full_version", return_value="v2.0.0")
    @patch(
        "twod_mcda.pipeline.find_neighbor_granules",
        return_value=(PREVIOUS, NEXT),
    )
    @patch("twod_mcda.pipeline.find_granule_file", return_value=CURRENT)
    def test_process_granule_invokes_refactored_implementation(
        self,
        find_granule_file,
        find_neighbor_granules,
        get_full_version,
        process_refactored,
    ):
        with patch(
            "twod_mcda.pipeline.PROCESSING_IMPLEMENTATION",
            "refactored",
        ):
            process_granule(config())

        process_refactored.assert_called_once()
        processing_config = process_refactored.call_args.args[0]
        self.assertEqual(
            processing_config["granule_date"],
            "2010-01-01T00-22-28ZN",
        )

    @patch("twod_mcda.pipeline.PROCESSING_IMPLEMENTATION", "invalid")
    def test_unknown_processing_implementation_is_rejected(self):
        from twod_mcda.pipeline import _run_processing

        with self.assertRaisesRegex(
            ValueError,
            "Unknown PROCESSING_IMPLEMENTATION",
        ):
            _run_processing({})


if __name__ == "__main__":
    unittest.main()
