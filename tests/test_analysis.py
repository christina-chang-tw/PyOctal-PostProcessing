"""
test_analysis.py

Regression tests for postprocessing.analysis.PAnalysis, built around a
synthetic periodic-Gaussian "loss" spectrum with a known resonance spacing.
Several of these exercise methods that previously raised AttributeError/
IndexError (xdata, fwhm, linewidth, get_3db_indices, resonances).

Run with: python -m unittest discover -s tests
"""
import unittest

import numpy as np
from scipy.signal import find_peaks

from postprocessing.analysis import PAnalysis


def make_spectrum(fsr=2e-9, span=(1540e-9, 1560e-9), n=40000,
                   sigma=0.15e-9, peak_height=30.0, target=1550e-9):
    """
    Build a synthetic spectrum with narrow Gaussian loss peaks spaced by
    `fsr`, one of which sits exactly at `target`.
    """
    x = np.linspace(*span, n)
    # Wrapped distance (in wavelength) to the nearest peak centre.
    phase = ((x - target + fsr / 2) % fsr) - fsr / 2
    y = peak_height * np.exp(-(phase ** 2) / (2 * sigma ** 2))
    return x, y


class TestPAnalysis(unittest.TestCase):
    def setUp(self):
        self.fsr_true = 2e-9
        self.target = 1550e-9
        self.cutoff = 10
        self.distance = 200
        self.x, self.y = make_spectrum(fsr=self.fsr_true, target=self.target)
        self.analysis = PAnalysis(
            self.x, self.y, self.target, cutoff=self.cutoff, distance=self.distance
        )

    def test_xdata_property_returns_input(self):
        # xdata was previously undefined -> AttributeError in resonances()/fwhm()/linewidth()
        np.testing.assert_array_equal(self.analysis.xdata, self.x)

    def test_peak_indices_match_independent_find_peaks(self):
        peaks, _ = find_peaks(self.y, distance=self.distance)
        peaks = peaks[self.y[peaks] - self.y.min() > self.cutoff]
        np.testing.assert_array_equal(self.analysis.peak_indices, peaks)

    def test_resonance_wavelength_near_target(self):
        self.assertAlmostEqual(
            self.analysis.resonance_wavelength, self.target, delta=self.fsr_true / 2
        )

    def test_fsr_matches_true_spacing(self):
        self.assertAlmostEqual(
            self.analysis.fsr(), self.fsr_true, delta=self.fsr_true * 0.01
        )

    def test_resonances_matches_resonances_indices(self):
        expected = self.x[self.analysis.resonances_indices(self.cutoff, self.distance)]
        np.testing.assert_array_equal(
            self.analysis.resonances(self.cutoff, self.distance), expected
        )

    def test_fwhm_positive_and_smaller_than_fsr(self):
        fwhm = self.analysis.fwhm()
        self.assertGreater(fwhm, 0)
        self.assertLess(fwhm, self.analysis.fsr())

    def test_linewidth_positive_and_same_order_as_fwhm(self):
        linewidth = self.analysis.linewidth()
        fwhm = self.analysis.fwhm()
        self.assertGreater(linewidth, 0)
        # Both measure the same ~3dB resonance width via different code
        # paths; they should land in the same ballpark.
        self.assertLess(linewidth / fwhm, 5)
        self.assertGreater(linewidth / fwhm, 0.2)

    def test_qfactor_is_resonance_over_fwhm(self):
        self.assertAlmostEqual(
            self.analysis.qfactor(),
            self.analysis.resonance_wavelength / self.analysis.fwhm(),
        )

    def test_get_3db_indices_bracket_resonance(self):
        # Previously broken: np.argmin() returns a scalar, indexing it with [-1]/[0] was a bug.
        left, right = self.analysis.get_3db_indices()
        self.assertLess(left, self.analysis.resonance_idx)
        self.assertGreater(right, self.analysis.resonance_idx)

    def test_centering_only_shifts_xdata(self):
        centered = self.analysis.centering(self.analysis.resonance_idx)
        np.testing.assert_allclose(np.diff(centered), np.diff(self.x))

    def test_get_range_idx_returns_ordered_bounds(self):
        idx_min, idx_max = self.analysis.get_range_idx(xrange=self.fsr_true / 4)
        self.assertLess(idx_min, idx_max)


if __name__ == "__main__":
    unittest.main()
