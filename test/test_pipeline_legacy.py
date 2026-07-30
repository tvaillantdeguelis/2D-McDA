from pathlib import Path
import unittest
from unittest.mock import patch

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
            "folder": "/data",
            "version": "5.00",
            "type": "Standard",
        },
        "processing": {
            "save_development_data": False,
            "process_up_to_40km": False,
        },
        "output": {
            "folder": "/output",
        },
        "slicing": {
            "type": "profindex",
            "start": None,
            "end": None,
        },
    }


class LegacyPipelineTests(unittest.TestCase):
    @patch("twod_mcda.pipeline.get_full_version", return_value="v2.0.0")
    def test_legacy_config_maps_paths_and_versions(self, get_full_version):
        legacy = _legacy_config(config(), CURRENT, PREVIOUS, NEXT)

        self.assertEqual(
            legacy["granule_date"],
            "2010-01-01T00-22-28ZN",
        )
        self.assertEqual(legacy["cal_lid_l1_version"], "V5.00")
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
        self.assertEqual(legacy["index30m_alt_max"], 600)

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


if __name__ == "__main__":
    unittest.main()
