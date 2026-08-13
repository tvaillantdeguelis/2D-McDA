import unittest

import numpy as np

from twod_mcda.algorithm.filtering import gaussian_2d_window, replace_maybe
from twod_mcda.algorithm.flags import (
    FLAG_AFA,
    FLAG_FA,
    FLAG_MAYBE,
    FLAG_NOTHING,
    FLAG_SMALL_STRIPS,
)


class Gaussian2DWindowTests(unittest.TestCase):
    def test_separable_convolution_matches_direct_weighted_average(self):
        data = np.arange(81, dtype=float).reshape(9, 9)
        mask = np.zeros(data.shape, dtype=bool)
        mask[3, 3] = True
        mask[4, 4] = True
        mask[5, 5] = True
        signal = np.ma.array(data, mask=mask)
        feature = np.full(data.shape, FLAG_NOTHING, dtype=np.uint8)
        feature[4, 4] = FLAG_FA

        actual, actual_sigma = gaussian_2d_window(
            3,
            1.5,
            signal,
            feature,
            ab_sigma=2.0,
            height_window=3,
            vertical_gauss_sigma=0.75,
        )

        horizontal = np.exp(-(np.arange(3) - 1) ** 2 / (2 * 1.5**2))
        vertical = np.exp(-(np.arange(3) - 1) ** 2 / (2 * 0.75**2))
        weights = np.outer(horizontal, vertical)
        expected = np.ma.masked_all(data.shape, dtype=float)
        for i in range(1, data.shape[0] - 1):
            for j in range(1, data.shape[1] - 1):
                if mask[i, j] and feature[i, j] != FLAG_FA:
                    continue
                window = signal[i - 1:i + 2, j - 1:j + 2]
                valid = ~np.ma.getmaskarray(window)
                if np.any(valid):
                    expected[i, j] = np.sum(
                        window.data[valid] * weights[valid]
                    ) / np.sum(weights[valid])

        np.testing.assert_array_equal(actual.mask, expected.mask)
        np.testing.assert_allclose(actual.compressed(), expected.compressed(), rtol=1e-14)
        self.assertAlmostEqual(actual_sigma, 2.0 / np.sqrt(np.sum(weights)))


class ReplaceMaybeTests(unittest.TestCase):
    def test_component_is_kept_at_size_limit(self):
        feature = np.zeros((4, 4), dtype=np.uint8)
        feature[1, 1:3] = FLAG_MAYBE

        actual = replace_maybe(2, feature, FLAG_DETECTION_LEVEL=2)

        expected = feature.copy()
        expected[1, 1:3] = 2
        np.testing.assert_array_equal(actual, expected)

    def test_component_is_removed_below_size_limit(self):
        feature = np.zeros((4, 4), dtype=np.uint8)
        feature[1, 1:3] = FLAG_MAYBE

        actual = replace_maybe(3, feature, FLAG_DETECTION_LEVEL=2)

        np.testing.assert_array_equal(actual, np.zeros((4, 4), dtype=np.uint8))

    def test_previous_detection_level_counts_towards_component_size(self):
        feature = np.zeros((4, 4), dtype=np.uint8)
        feature[1, 0:2] = 2
        feature[1, 2] = FLAG_MAYBE

        with_previous = replace_maybe(3, feature, FLAG_DETECTION_LEVEL=3)
        without_previous = replace_maybe(
            3, feature, FLAG_DETECTION_LEVEL=3, prev_detect=False
        )

        self.assertEqual(with_previous[1, 2], 3)
        self.assertEqual(without_previous[1, 2], FLAG_NOTHING)

    def test_second_previous_detection_level_can_count_towards_size(self):
        feature = np.zeros((4, 4), dtype=np.uint8)
        feature[1, 0:2] = 2
        feature[1, 2] = FLAG_MAYBE

        with_second_previous = replace_maybe(
            3,
            feature,
            FLAG_DETECTION_LEVEL=4,
            prev_detect=False,
            prevprev_detect=True,
        )

        self.assertEqual(with_second_previous[1, 2], 4)

    def test_current_detection_connects_across_low_confidence_flags(self):
        for low_confidence_flag in (FLAG_FA, FLAG_AFA, FLAG_SMALL_STRIPS):
            with self.subTest(low_confidence_flag=low_confidence_flag):
                feature = np.zeros((4, 3), dtype=np.uint8)
                feature[0, 1] = 3
                feature[1, 1] = low_confidence_flag
                feature[2, 1] = FLAG_MAYBE

                actual = replace_maybe(2, feature, FLAG_DETECTION_LEVEL=3)

                self.assertEqual(actual[2, 1], 3)

    def test_mask_is_preserved_and_masked_maybe_is_not_classified(self):
        data = np.zeros((3, 3), dtype=np.uint8)
        data[1, 1] = FLAG_MAYBE
        feature = np.ma.array(data, mask=np.eye(3, dtype=bool))

        actual = replace_maybe(1, feature, FLAG_DETECTION_LEVEL=2)

        np.testing.assert_array_equal(actual.mask, feature.mask)
        self.assertTrue(actual.mask[1, 1])


if __name__ == "__main__":
    unittest.main()
