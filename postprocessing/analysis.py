"""
analysis.py

Perform photonics simulation/experimental data analysis
"""
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from postprocessing.utils.conversion import db2w

class PAnalysis:
    """
    This is for performing analysis based on the transmission spectrum.

    Parameters:
        xdata (np.ndarray): The x-axis data. [dB]
        ydata (np.ndarray): The y-axis data. [dB]
        wavelength (float): The wavelength of interest in [m].
    """
    def __init__(self, xdata: np.array, ydata: np.array, wavelength: float, cutoff: float=20, distance: int=100) -> None:
        self._xdata = xdata
        self._ydata = np.absolute(ydata)
        self.target_wavelength = wavelength
        self.target_wavelength_idx = np.argmin(np.abs(xdata - wavelength))
        self._peak_indices = self.resonances_indices(cutoff=cutoff, distance=distance)

    def sanity_check(self, xlim: list=None, ylim: list=None) -> None:
        """
        Check your human brain sanity.

        If not only peaks are shown, then play around with the cutoff and distance.
        """
        plt.clf()
        plt.plot(self._xdata*1E+09, self._ydata)
        plt.scatter(self._xdata[self._peak_indices]*1E+09, self._ydata[self._peak_indices], marker="x")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Transmission [dB]")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()

    @property
    def xdata(self) -> np.ndarray:
        return self._xdata

    @property
    def ydata(self) -> np.ndarray:
        return self._ydata

    @property
    def peak_indices(self) -> np.ndarray:
        return self._peak_indices

    def __peak_idx(self) -> int:
        """
        Get the index that is closest to the target wavelength in the peak.
        """
        normalised = self._peak_indices - self.target_wavelength_idx
        return np.argmin(np.abs(normalised)) # this returns the index of the minimised term in peak_indices

    @property
    def off_resonance_idx(self) -> int:
        """
        Get the power maximum index that is closest to the target wavelength in the orriginal data.
        """
        if self.resonance_idx < self.target_wavelength_idx:
            return (self.resonance_idx + self._peak_indices[self.__peak_idx() + 1]) // 2

        return (self.resonance_idx + self._peak_indices[self.__peak_idx() - 1]) // 2

    @property
    def resonance_idx(self) -> int:
        """
        Get the power maximum index that is closest to the target wavelength in the original data.
        """
        return self._peak_indices[self.__peak_idx()]

    @property
    def resonance_wavelength(self) -> float:
        """
        Get the resonance wavelength that is closest to the target wavelength in the orriginal data.
        """
        return self._xdata[self.resonance_idx]

    def get_3db_indices(self) -> Tuple[int, int]:
        """
        Get the the two 3dB index closest to the resonance.
        """
        left_idx = np.argmin(np.abs(self._ydata[:self.resonance_idx] - 3))
        right_idx = np.argmin(np.abs(self._ydata[self.resonance_idx:] - 3)) + self.resonance_idx
        return left_idx, right_idx

    def get_range_idx(self, xrange: int=1E-09) -> np.ndarray:
        """
        Get the range of the data closest to the resonance peak that is closest to the target
        wavelength. Return two indices that correspond to the range which encloses one peak.
        idx_low corresponds to lower frequency/higher wavelength.

        Parameters:
            range (int): The range of the data to be considered. int=1 means 1nm range.
        """
        res = self.resonance_wavelength
        idx_min = np.argmin(np.abs(self._xdata - (res - xrange/2)))
        idx_max = np.argmin(np.abs(self._xdata - (res + xrange/2)))

        return idx_min, idx_max

    def centering(self, idx: int) -> np.ndarray:
        """
        Center the x-axis data with respect to the resonance frequency.

        Parameters:
            idx (int): The index of the resonance frequency.

        Returns:
            np.ndarray: The centered x-axis data.
        """
        lmax_idx = np.argmax(self._ydata[:self.resonance_idx])
        rmax_idx = np.argmax(self._ydata[self.resonance_idx:]) + self.resonance_idx
        min_idx0 = np.argmin(self._ydata[lmax_idx:rmax_idx])
        xdata = self._xdata - self._xdata[lmax_idx + min_idx0]

        return xdata

    def resonances_indices(self, cutoff: float, distance: int) -> np.ndarray:
        """
        Find the peaks in the spectrum.

        Parameters:
            cutoff (float): The cutoff value for the peaks.
            distance (int): The minimum distance between peaks.

        Returns:
            np.ndarray: The indices of the peaks.
        """
        peaks, _ = find_peaks(self._ydata, distance=distance) # indices of the peaks

        # perform another filtering
        peaks = peaks[self._ydata[peaks] - min(self._ydata) > cutoff]
        return peaks

    def resonances(self, cutoff: float, distance: int) -> np.ndarray:
        """
        Find the peaks in the spectrum.

        Parameters:
            cutoff (float): The cutoff value for the peaks.
            distance (int): The minimum distance between peaks.

        Returns:
            np.ndarray: The wavelengths of the peaks.
        """
        return self._xdata[self.resonances_indices(cutoff, distance)]

    def _peaks_idx_for_averaging(self, num: int) -> np.ndarray:
        """
        Find the peaks for averaging.

        Parameters:
            num (int): The number of peaks.

        Returns:
            np.ndarray: The indices of the peaks for averaging.
        """
        if num >= len(self.peak_indices):
            return self.peak_indices

        target_idx = np.argmin(np.abs(self.peak_indices - self.target_wavelength_idx))
        peaks_for_avg = self.peak_indices[target_idx-num//2:target_idx+1+num//2]

        return peaks_for_avg

    def _peak_midpoints(self, num_peaks: int = 3) -> np.ndarray:
        """
        Get the indices midway between each pair of neighbouring peaks used
        for averaging. Used to bound the window around the resonance for
        linewidth/FWHM calculations.

        Parameters:
            num_peaks (int): The number of peaks to average over.

        Returns:
            np.ndarray: The midpoint indices between consecutive peaks.
        """
        peaks_idx = self._peaks_idx_for_averaging(num_peaks)
        return (peaks_idx[:-1] + peaks_idx[1:]) // 2

    def fsr(self, num_peaks: int=3) -> float:
        """
        Calculate the free spectral range of the resonator.


        Returns:
            float: The free spectral range of the resonator.
        """
        # find the closest peaks to the resonance wavelength
        if num_peaks < 2:
            raise ValueError("Number of peaks for averaging must be greater than 1.")

        peaks_idx = self._peaks_idx_for_averaging(num_peaks)

        # takes averaging
        fsr = 0
        for i in range(1, len(peaks_idx)):
            fsr += self._xdata[peaks_idx[i]] - self._xdata[peaks_idx[i-1]]

        return fsr / (len(peaks_idx)-1)

    def linewidth(self) -> float:
        """
        Calculate the linewidth of the resonator based on the 3dB points
        either side of resonance, taken directly from the dB-scale ydata.

        Returns:
            float: The linewidth of the resonator.
        """
        ydata = self._ydata

        if len(self.peak_indices) > 1:
            midpoints = self._peak_midpoints()
            ydata = ydata[midpoints[0]:midpoints[-1]]
            indices = np.where(ydata >= 3)[0] + midpoints[0]
        else:
            indices = np.where(ydata >= 3)[0]

        return np.absolute(self._xdata[indices[-1]] - self._xdata[indices[0]])

    def fwhm(self) -> float:
        """
        Calculate the full width at half maximum of the resonator, using the
        linear-power representation of the spectrum.

        Returns:
            float: The full width at half maximum.
        """
        # Convert ydata from dB to linear ratio
        tol = 0.01
        ydata = db2w(-(self._ydata - min(self._ydata)))

        midpoints = self._peak_midpoints()
        ydata = ydata[midpoints[0]:midpoints[-1]]
        half_indices = np.where(np.isclose(ydata, 0.5, atol=tol))[0] + midpoints[0]

        return np.absolute(self._xdata[half_indices[-1]] - self._xdata[half_indices[0]])

    def qfactor(self) -> float:
        """
        Calculate the quality factor of the resonator.

        Returns:
            float: The quality factor of the resonator.
        """
        return self.resonance_wavelength/self.fwhm()


def main():
    filename = "output/Radius1_t17_3v.csv"
    data = pd.read_csv(filename)
    target_wavelength = 1550e-09
    wavelength = data["Wavelength"].values
    ydata = data["Loss [dB]"].values

    analysis = PAnalysis(wavelength, ydata, target_wavelength, cutoff=10, distance=100)

    print(f"Free spectral range [nm]: {np.round(analysis.fsr()*1e+09, 3)}")
    print(f"Full width half maximum [nm]: {np.round(analysis.fwhm()*1e+09, 3)}")
    print(f"Quality factor: {np.round(analysis.qfactor(), 3)}")

    peaks = analysis.peak_indices
    plt.plot(wavelength*1e+09, ydata, label="data")
    plt.plot(wavelength[peaks]*1e+09, ydata[peaks], "x", label="peaks")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Loss [dB]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
