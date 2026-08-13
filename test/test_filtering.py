import unittest

import numpy as np

from twod_mcda.algorithm.filtering import gaussian_2d_window
from twod_mcda.algorithm.flags import FLAG_FA, FLAG_NOTHING


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


if __name__ == "__main__":
    unittest.main()
