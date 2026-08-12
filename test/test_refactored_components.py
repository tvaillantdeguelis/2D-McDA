import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from netCDF4 import Dataset
import numpy as np

from twod_mcda.io.product_writer import (
    build_output_variables,
    output_filename,
    write_product,
)
from twod_mcda.preprocessing.neighbors import append_adjacent_profiles
from twod_mcda.processing.granule_processor import _store_development
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

    def test_development_slices_share_the_full_profile_dimension(self):
        output = {}
        first = {
            "transmittance": np.ones((3, 2), dtype=np.float32),
            "steps": np.ones((2, 3, 2), dtype=np.uint8),
        }
        second = {
            "transmittance": np.full((3, 2), 2.0, dtype=np.float32),
            "steps": np.full((2, 3, 2), 2, dtype=np.uint8),
        }

        _store_development(output, first, 10, 12, 10, 5)
        _store_development(output, second, 12, 14, 10, 5)

        self.assertEqual(output["transmittance"].shape, (5, 2))
        self.assertEqual(output["steps"].shape, (2, 5, 2))
        np.testing.assert_array_equal(
            output["transmittance"][:, 0],
            np.array([1.0, 1.0, 2.0, 2.0, 2.0]),
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
            {variable.name for variable in variables.values()},
            {
                "Trajectory_ID",
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
            "2010-01-01T00-22-28ZN.nc",
        )

    def test_product_is_cf_oriented_netcdf4(self):
        arrays = {
            "Profile_ID": np.array([1, 2], dtype=np.int32),
            "Profile_Time": np.array([536457600.0, 536457601.0]),
            "Profile_UTC_Time": np.array([100101.0, 100101.00001]),
            "Latitude": np.array([1.0, 1.1], dtype=np.float32),
            "Longitude": np.array([2.0, 2.1], dtype=np.float32),
            "Parallel_Detection_Flags_532": np.zeros((2, 2), dtype=np.uint8),
            "Perpendicular_Detection_Flags_532": np.zeros((2, 2), dtype=np.uint8),
            "Detection_Flags_1064": np.zeros((2, 2), dtype=np.uint8),
            "Composite_Detection_Flags": np.zeros((2, 2), dtype=np.uint8),
        }
        result = ProcessingResult(
            data=arrays,
            development={},
            altitude=np.array([0.0, 1.0], dtype=np.float32),
            longitude_min=2.0,
            longitude_max=2.1,
        )

        with TemporaryDirectory() as directory:
            product_request = request()
            object.__setattr__(
                product_request,
                "output_directory",
                Path(directory),
            )
            path = write_product(product_request, result)

            with Dataset(path) as dataset:
                self.assertEqual(dataset.data_model, "NETCDF4")
                self.assertEqual(dataset.Conventions, "CF-1.13")
                self.assertEqual(dataset.featureType, "trajectoryProfile")
                self.assertEqual(
                    dataset.variables["Trajectory_ID"].cf_role,
                    "trajectory_id",
                )
                self.assertEqual(
                    dataset.variables["Profile_Time"].calendar,
                    "tai",
                )
                self.assertEqual(
                    dataset.variables["Latitude"].units,
                    "degrees_north",
                )
                self.assertEqual(
                    dataset.variables["Altitude"].positive,
                    "up",
                )
                filters = dataset.variables[
                    "Composite_Detection_Flags"
                ].filters()
                self.assertTrue(filters["zlib"])


if __name__ == "__main__":
    unittest.main()
