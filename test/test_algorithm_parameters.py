import unittest

from twod_mcda.algorithm.parameters import (
    FeatureDetectionParameters,
    SurfaceDetectionParameters,
    get_feature_detection_coef,
)


class AlgorithmParameterTests(unittest.TestCase):
    def test_surface_window_depends_on_channel(self):
        self.assertEqual(SurfaceDetectionParameters("532_par").N, 2)
        self.assertEqual(SurfaceDetectionParameters("532_per").N, 2)
        self.assertEqual(SurfaceDetectionParameters("1064").N, 4)

    def test_feature_coefficients_preserve_operational_values(self):
        self.assertEqual(
            get_feature_detection_coef("532_par", 4),
            (1, 100000, (17, 105), (105, 35)),
        )
        self.assertEqual(
            get_feature_detection_coef("532_per", 4),
            (1, 1000, (9, 51), (15, 5)),
        )
        self.assertEqual(
            get_feature_detection_coef("1064", 0),
            (None, None, None, None),
        )

    def test_invalid_channels_raise_value_error(self):
        with self.assertRaises(ValueError):
            SurfaceDetectionParameters("invalid")
        with self.assertRaises(ValueError):
            FeatureDetectionParameters("invalid")
        with self.assertRaises(ValueError):
            get_feature_detection_coef("invalid", 0)


if __name__ == "__main__":
    unittest.main()
