from dataclasses import replace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from netCDF4 import Dataset
import numpy as np

from twod_mcda.caliop.reader import CALIPSOReader
from twod_mcda.output.product import (
    build_output_variables,
    output_filename,
    write_product,
)
from twod_mcda.pipeline import (
    DETECTION_MASKS,
    PROFILE_METADATA,
    _empty_output,
    _open_inputs,
    _slice_description,
    _store_development,
    _store_slice,
)
from twod_mcda.workflow.neighbors import append_adjacent_profiles
from twod_mcda.workflow.models import (
    ProcessingRequest,
    ProcessingResult,
    SliceData,
)
from twod_mcda.workflow.processing import _trim_neighbor_profiles
from twod_mcda.workflow.slicing import plan_slices


def request():
    return ProcessingRequest(
        granule_date="2010-01-01T00-22-28ZN",
        caliop_version="V5.00",
        current_directory=Path("/data/current"),
        previous_granule=None,
        previous_directory=None,
        next_granule=None,
        next_directory=None,
        subset_active=True,
        subset_mode="profindex",
        subset_start=0,
        subset_end=None,
        save_development_data=False,
        output_version="V2.0.0",
        output_product_type="Dev",
        output_directory=Path("/output"),
        maximum_altitude_km=30,
        maximum_altitude_index=600,
    )


class RefactoredComponentTests(unittest.TestCase):
    def test_slice_plan_separates_result_profiles_from_context(self):
        (
            profile_starts,
            profile_ends,
            context_starts,
            context_ends,
        ) = plan_slices(0, 10000, 3000, 250)

        np.testing.assert_array_equal(
            profile_starts,
            np.array([0, 3000, 6000, 9000]),
        )
        np.testing.assert_array_equal(
            profile_ends,
            np.array([3000, 6000, 9000, 10000]),
        )
        np.testing.assert_array_equal(
            context_starts,
            np.array([-250, 2750, 5750, 8750]),
        )
        np.testing.assert_array_equal(
            context_ends,
            np.array([3250, 6250, 9250, 10250]),
        )

    def test_slice_plan_handles_a_single_profile(self):
        result = plan_slices(42, 42, 3000, 250)

        expected = (
            np.array([42]),
            np.array([42]),
            np.array([-208]),
            np.array([292]),
        )
        for actual, expected_values in zip(result, expected):
            np.testing.assert_array_equal(actual, expected_values)

    def test_slice_descriptions_show_result_and_context_bounds(self):
        bounds = (
            (0, 3000, -250, 3250),
            (3000, 6000, 2750, 6250),
            (6000, 9000, 5750, 9250),
            (9000, 10000, 8750, 10250),
        )

        descriptions = [
            _slice_description(index, 4, *slice_bounds)
            for index, slice_bounds in enumerate(bounds, start=1)
        ]

        self.assertEqual(
            descriptions,
            [
                "Process slice 1/4 (profiles 0 to 3000 using slice -250 to 3250)",
                "Process slice 2/4 (profiles 3000 to 6000 using slice 2750 to 6250)",
                "Process slice 3/4 (profiles 6000 to 9000 using slice 5750 to 9250)",
                "Process slice 4/4 (profiles 9000 to 10000 using slice 8750 to 10250)",
            ],
        )

    @patch("twod_mcda.caliop.reader.HDF4Reader")
    def test_caliop_reader_reads_only_the_requested_hyperslab(self, hdf_reader):
        hdf = hdf_reader.return_value.__enter__.return_value
        hdf.get_sds_keys.return_value = {
            "Latitude": (0, (10,), 5, 0),
            "Science": (1, (10, 2), 5, 0),
            "Altitude": (2, (2,), 5, 0),
        }
        hdf.get_metadata_keys.return_value = ("Lidar_Data_Altitudes",)
        hdf.get_metadata.return_value = [0.0, 1.0]
        hdf.get_fillvalue.return_value = -9999.0
        hdf.get_data.return_value = np.arange(6).reshape(3, 2)
        reader = CALIPSOReader("granule.hdf")

        first = reader.get_data("Science", 2, 4)
        second = reader.get_data("Science", 2, 4)
        altitude = reader.get_data("Lidar_Data_Altitudes")

        np.testing.assert_array_equal(first, np.arange(6).reshape(3, 2))
        np.testing.assert_array_equal(second, first)
        self.assertIsInstance(altitude, np.ma.MaskedArray)
        np.testing.assert_array_equal(altitude, np.array([0.0, 1.0]))
        hdf.get_data.assert_called_once_with(
            "Science",
            start=[2, 0],
            count=[3, 2],
            do_squeeze=False,
        )
        reader.close()

    @patch("twod_mcda.pipeline.open_granule")
    def test_interior_subset_does_not_load_neighbors(self, open_granule):
        current_reader = SimpleNamespace(
            filepath="/data/current.hdf",
            prof_min=4000,
            prof_max=5000,
            data_reader=SimpleNamespace(nb_profiles=10000),
        )
        open_granule.return_value = current_reader
        stack = MagicMock()
        stack.enter_context.side_effect = lambda reader: reader
        processing_request = replace(
            request(),
            previous_granule="previous",
            next_granule="next",
            subset_start=4000,
            subset_end=5000,
        )

        result = _open_inputs(processing_request, stack)

        open_granule.assert_called_once()
        self.assertIs(result[0], current_reader)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        np.testing.assert_array_equal(result[5], np.array([4000]))
        np.testing.assert_array_equal(result[6], np.array([5000]))
        np.testing.assert_array_equal(result[7], np.array([3750]))
        np.testing.assert_array_equal(result[8], np.array([5250]))

    @patch("twod_mcda.pipeline.open_granule")
    def test_context_touching_granule_edges_does_not_load_neighbors(
        self,
        open_granule,
    ):
        stack = MagicMock()
        stack.enter_context.side_effect = lambda reader: reader

        cases = (
            (250, 1000, np.array([0]), np.array([1250])),
            (9000, 9749, np.array([8750]), np.array([9999])),
        )
        for subset_start, subset_end, expected_starts, expected_ends in cases:
            with self.subTest(start=subset_start, end=subset_end):
                current_reader = SimpleNamespace(
                    filepath="/data/current.hdf",
                    prof_min=subset_start,
                    prof_max=subset_end,
                    data_reader=SimpleNamespace(nb_profiles=10000),
                )
                open_granule.reset_mock()
                open_granule.return_value = current_reader
                processing_request = replace(
                    request(),
                    previous_granule="previous",
                    next_granule="next",
                    subset_start=subset_start,
                    subset_end=subset_end,
                )

                result = _open_inputs(processing_request, stack)

                open_granule.assert_called_once()
                self.assertIsNone(result[1])
                self.assertIsNone(result[2])
                np.testing.assert_array_equal(result[7], expected_starts)
                np.testing.assert_array_equal(result[8], expected_ends)

    @patch("twod_mcda.pipeline.read_slice", return_value={})
    @patch("twod_mcda.pipeline.open_granule")
    def test_subset_at_start_loads_previous_context(
        self,
        open_granule,
        read_slice,
    ):
        current_reader = SimpleNamespace(
            filepath="/data/current.hdf",
            prof_min=0,
            prof_max=4000,
            data_reader=SimpleNamespace(nb_profiles=10000),
        )
        previous_reader = SimpleNamespace(
            filepath="/data/previous.hdf",
            prof_min=9750,
            prof_max=9999,
        )
        open_granule.side_effect = [current_reader, previous_reader]
        stack = MagicMock()
        stack.enter_context.side_effect = lambda reader: reader
        processing_request = replace(
            request(),
            previous_granule="previous",
            subset_start=0,
            subset_end=4000,
        )

        result = _open_inputs(processing_request, stack)

        self.assertEqual(open_granule.call_count, 2)
        read_slice.assert_called_once_with(previous_reader, 9750, 9999)
        np.testing.assert_array_equal(result[5], np.array([0, 3000]))
        np.testing.assert_array_equal(result[6], np.array([3000, 4000]))
        np.testing.assert_array_equal(result[7], np.array([-250, 2750]))
        np.testing.assert_array_equal(result[8], np.array([3250, 4250]))

    @patch("twod_mcda.pipeline.read_slice", return_value={})
    @patch("twod_mcda.pipeline.open_granule")
    def test_subset_near_start_loads_only_the_missing_context(
        self,
        open_granule,
        read_slice,
    ):
        current_reader = SimpleNamespace(
            filepath="/data/current.hdf",
            prof_min=100,
            prof_max=1000,
            data_reader=SimpleNamespace(nb_profiles=10000),
        )
        previous_reader = SimpleNamespace(
            filepath="/data/previous.hdf",
            prof_min=9850,
            prof_max=9999,
        )
        open_granule.side_effect = [current_reader, previous_reader]
        stack = MagicMock()
        stack.enter_context.side_effect = lambda reader: reader
        processing_request = replace(
            request(),
            previous_granule="previous",
            subset_start=100,
            subset_end=1000,
        )

        result = _open_inputs(processing_request, stack)

        self.assertEqual(open_granule.call_args_list[1].args[3:], (-150, None))
        read_slice.assert_called_once_with(previous_reader, 9850, 9999)
        np.testing.assert_array_equal(result[7], np.array([-150]))
        np.testing.assert_array_equal(result[8], np.array([1250]))

    @patch("twod_mcda.pipeline.read_slice", return_value={})
    @patch("twod_mcda.pipeline.open_granule")
    def test_full_granule_loads_and_returns_both_neighbors(
        self,
        open_granule,
        read_slice,
    ):
        current_reader = SimpleNamespace(
            filepath="/data/current.hdf",
            prof_min=0,
            prof_max=9999,
            data_reader=SimpleNamespace(nb_profiles=10000),
        )
        previous_reader = SimpleNamespace(
            filepath="/data/previous.hdf",
            prof_min=9750,
            prof_max=9999,
        )
        next_reader = SimpleNamespace(
            filepath="/data/next.hdf",
            prof_min=0,
            prof_max=249,
        )
        open_granule.side_effect = [current_reader, previous_reader, next_reader]
        stack = MagicMock()
        stack.enter_context.side_effect = lambda reader: reader
        processing_request = replace(
            request(),
            previous_granule="previous",
            next_granule="next",
        )

        result = _open_inputs(processing_request, stack)

        self.assertEqual(open_granule.call_count, 3)
        self.assertEqual(read_slice.call_count, 2)
        self.assertIs(result[0], current_reader)
        self.assertIs(result[1], previous_reader)
        self.assertIs(result[2], next_reader)

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

    def test_neighbor_context_is_trimmed_using_the_actual_counts(self):
        slice_data = SliceData(
            input={
                "Profile_Time": np.arange(6),
                "Lidar_Data_Altitudes": np.arange(4),
            },
            masks={"mask": np.arange(6)[:, np.newaxis]},
            development={
                "steps": np.arange(12).reshape(2, 6, 1),
            },
            previous_context_count=2,
            next_context_count=1,
        )

        _trim_neighbor_profiles(slice_data)

        np.testing.assert_array_equal(
            slice_data.input["Profile_Time"],
            np.array([2, 3, 4]),
        )
        np.testing.assert_array_equal(
            slice_data.input["Lidar_Data_Altitudes"],
            np.arange(4),
        )
        np.testing.assert_array_equal(
            slice_data.masks["mask"][:, 0],
            np.array([2, 3, 4]),
        )
        np.testing.assert_array_equal(
            slice_data.development["steps"][:, :, 0],
            np.array([[2, 3, 4], [8, 9, 10]]),
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

        _store_development(output, first, 10, 12, 10, 10, 5)
        _store_development(output, second, 12, 14, 12, 10, 5)

        self.assertEqual(output["transmittance"].shape, (5, 2))
        self.assertEqual(output["steps"].shape, (2, 5, 2))
        np.testing.assert_array_equal(
            output["transmittance"][:, 0],
            np.array([1.0, 1.0, 2.0, 2.0, 2.0]),
        )

    def test_storing_a_slice_excludes_its_context(self):
        profile_min = 40000
        profile_max = 43000
        context_min = 39750
        context_max = 43250
        requested_min = 40000
        profile_indexes = np.arange(context_min, context_max + 1)
        output = _empty_output(10001, 1)
        slice_data = SliceData(
            input={name: profile_indexes.copy() for name in PROFILE_METADATA},
            masks={
                name: profile_indexes[:, np.newaxis].astype(np.uint8)
                for name in DETECTION_MASKS
            },
        )

        _store_slice(
            output,
            slice_data,
            profile_min,
            profile_max,
            context_min,
            requested_min,
        )

        np.testing.assert_array_equal(
            output["Profile_ID"][:3001],
            np.arange(40000, 43001),
        )
        self.assertEqual(output["Profile_ID"].shape, (10001,))

    def test_development_data_is_cropped_to_the_requested_subset(self):
        output = {}
        values = np.arange(3501)[:, np.newaxis]

        _store_development(
            output,
            {"values": values},
            40000,
            43000,
            39750,
            40000,
            10001,
        )

        self.assertEqual(output["values"].shape, (10001, 1))
        np.testing.assert_array_equal(
            output["values"][:3001, 0],
            np.arange(250, 3251),
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
        self.assertEqual(
            output_filename(replace(request(), subset_start=None), result),
            "CAL_LID_L2_2D_McDA-Dev-V2-0-0."
            "2010-01-01T00-22-28ZN.nc",
        )
        self.assertEqual(
            output_filename(
                replace(
                    request(),
                    subset_active=False,
                    subset_mode="longitude",
                    subset_start=63.2,
                    subset_end=61.3,
                ),
                result,
            ),
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
