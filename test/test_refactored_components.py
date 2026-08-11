import unittest

import numpy as np

from twod_mcda.io.product_writer import build_output_variables, output_filename
from twod_mcda.preprocessing.neighbors import append_adjacent_profiles
from twod_mcda.processing.models import ProcessingRequest, ProcessingResult


def request():
    return ProcessingRequest.from_mapping(
        {
            "granule_date": "2010-01-01T00-22-28ZN",
            "cal_lid_l1_version": "V5.00",
            "cal_lid_l1_type": "Standard",
            "folder_path": "/data/current",
            "previous_granule": None,
            "previous_folder_path": None,
            "next_granule": None,
            "next_folder_path": None,
            "slice_type": "profindex",
            "slice_start": 0,
            "slice_end": None,
            "save_development_data": False,
            "version_2d_mcda": "V2.0.0",
            "type_2d_mcda": "Dev",
            "out_folder": "/output",
            "index30m_alt_max": 600,
        }
    )


class RefactoredComponentTests(unittest.TestCase):
    def test_append_adjacent_profiles_keeps_altitude_unchanged(self):
        current = {
            "Profile_Time": np.array([2.0, 3.0]),
            "Lidar_Data_Altitudes": np.array([0.0, 1.0]),
        }
        previous = {
            "Profile_Time": np.array([0.0, 1.0]),
            "Lidar_Data_Altitudes": np.array([0.0, 1.0]),
        }

        result = append_adjacent_profiles(current, previous, "start")

        np.testing.assert_array_equal(
            result["Profile_Time"],
            np.array([0.0, 1.0, 2.0, 3.0]),
        )
        self.assertIs(
            result["Lidar_Data_Altitudes"],
            current["Lidar_Data_Altitudes"],
        )

    def test_output_schema_matches_current_product(self):
        arrays = {
            "Profile_ID": np.array([1], dtype=np.int32),
            "Profile_Time": np.array([1.0]),
            "Profile_UTC_Time": np.array([1.0]),
            "Latitude": np.array([1.0], dtype=np.float32),
            "Longitude": np.array([2.0], dtype=np.float32),
            "Parallel_Detection_Flags_532": np.zeros((1, 2), dtype=np.uint8),
            "Perpendicular_Detection_Flags_532": np.zeros((1, 2), dtype=np.uint8),
            "Detection_Flags_1064": np.zeros((1, 2), dtype=np.uint8),
            "Composite_Detection_Flags": np.zeros((1, 2), dtype=np.uint8),
        }
        result = ProcessingResult(
            data=arrays,
            development={},
            altitude=np.array([0.0, 1.0]),
            longitude_min=2.0,
            longitude_max=2.0,
        )

        variables = build_output_variables(result, False)

        self.assertEqual(
            {variable.key for variable in variables.values()},
            {
                "Profile_ID",
                "Profile_Time",
                "Profile_UTC_Time",
                "Latitude",
                "Longitude",
                "Altitude",
                "Parallel_Detection_Flags_532",
                "Perpendicular_Detection_Flags_532",
                "Detection_Flags_1064",
                "Composite_Detection_Flags",
            },
        )
        self.assertEqual(
            output_filename(request(), result),
            "CAL_LID_L2_2D_McDA-Dev-V2-0-0."
            "2010-01-01T00-22-28ZN.hdf",
        )


if __name__ == "__main__":
    unittest.main()
