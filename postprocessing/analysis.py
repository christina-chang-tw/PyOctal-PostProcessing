"""
analysis.py

Perform photonics simulation/experimental data analysis
"""
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
    def __init__(self, xdata: np.array, ydata: np.array, wavelength: float, cutoff: float=20, distance: int=100):
        self._xdata = xdata
        self._ydata = np.absolute(ydata)
        self.wavelength = wavelength
        self.wavelength_idx = np.argmin(np.abs(xdata - wavelength))
        self._peaks = self.resonances(cutoff=cutoff, distance=distance)

    def sanity_check(self, xlim: list=None, ylim: list=None):
        """
        Check your human brain sanity.
        
        If not only peaks are shown, then play around with the cutoff and distance.
        """
        plt.clf()
        plt.plot(self._xdata*1E+09, self._ydata)
        plt.scatter(self._xdata[self._peaks]*1E+09, self._ydata[self._peaks], marker="x")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Transmission [dB]")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
        
    @property
    def xdata(self):
        return self._xdata
    
    @property
    def ydata(self):
        return self._ydata

    @property
    def peaks(self):
        return self._peaks

    def __true_peak_idx(self):
        """ 
        Get the index that is closest to the target wavelength in the peak. """
        return np.argmin(np.abs(self._peaks - self.wavelength_idx))
    
    @property
    def true_offres_idx(self):
        """ 
        Get the maximum power index (true resonance) that is closest to the target wavelength in the orriginal data. 
        """
        if self.true_res_idx < self.wavelength_idx:
            return (self._peaks[self.__true_peak_idx()] + self._peaks[self.__true_peak_idx() + 1]) // 2

        return (self._peaks[self.__true_peak_idx()] + self._peaks[self.__true_peak_idx() - 1]) // 2

    @property
    def true_res_idx(self):
        """ 
        Get the resonance index that is closest to the target wavelength in the orriginal data. 
        """
        return self._peaks[self.__true_peak_idx()]
    
    @property
    def true_res_wavelength(self):
        """ 
        Get the resonance wavelength that is closest to the target wavelength in the orriginal data. 
        """
        return self._xdata[self.true_res_idx]
    
    def get_3db_indices(self) -> Tuple:
        """
        Get the the two 3dB index closest to the resonance.
        """
        return np.argmin(np.abs(self._ydata[:self.true_res_idx] - 3))[-1], \
            np.argmin(np.abs(self._ydata[self.true_res_idx:] - 3))[0]

    def get_range_idx(self, xrange: int=1E-09):
        """
        Get the range of the data closest to the resonance peak that is closest to the target
        wavelength. Return two indices that correspond to the range which encloses one peak.
        idx_low corresponds to lower frequency/higher wavelength.

        Parameters:
            range (int): The range of the data to be considered. int=1 means 1nm range.
        """
        res = self.true_res_wavelength
        idx_min = np.argmin(np.abs(self._xdata - (res - xrange/2)))
        idx_max = np.argmin(np.abs(self._xdata - (res + xrange/2)))

        return idx_min, idx_max

    def centering(self) -> np.ndarray:
        """
        Center the x-axis data with respect to the resonance frequency.

        Parameters:
            idx (int): The index of the resonance frequency.

        Returns:
            np.ndarray: The centered x-axis data.
        """
        lmax_idx = np.argmax(self._ydata[:self.true_res_idx])
        rmax_idx = np.argmax(self._ydata[self.true_res_idx:]) + self.true_res_idx
        min_idx0 = np.argmin(self._ydata[lmax_idx:rmax_idx])
        xdata = self._xdata - self._xdata[lmax_idx + min_idx0]
        return xdata

    def modeff_phase(self, voltage: np.array, phase: np.array, length: float) -> np.array:
        """
        Modulation efficiency Vm.
        """
        return length*voltage*np.pi / phase


    def resonances(self, cutoff: float, distance: int) -> list:
        """
        Find the peaks in the spectrum.

        Parameters:
            cutoff (float): The cutoff value for the peaks.
            distance (int): The minimum distance between peaks.

        Returns:
            list: The indices of the peaks.
        """
        peaks, _ = find_peaks(self._ydata, distance=distance)

        # perform another filtering
        peaks = peaks[self._ydata[peaks] - min(self._ydata) > cutoff]
        return peaks
   
    def _peaks_idx_for_averaging(self, num: int) -> list:
        """
        Find the peaks for averaging.

        Parameters:
            num (int): The number of peaks.

        Returns:
            list: The indices of the peaks for averaging.
        """
        if num >= len(self.peaks):
            return self.peaks

        target_idx = np.argmin(np.abs(self.peaks - self.wavelength_idx))
        peaks_for_avg = self.peaks[target_idx-num//2:target_idx+1+num//2]

        return peaks_for_avg

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
        
        return fsr/(len(peaks_idx)-1)
    
    def linewidth(self) -> float:
        """
        Calculate the linewidth of the resonator.
        Must provide linear scale ydata

        Returns:
            float: The linewidth of the resonator.
        """
        peaks = self._peaks
        xdata = self._xdata
        ydata = self._ydata

        if len(peaks) > 1:
            peaks = self._peaks_idx_for_averaging(3)
            peak_midpoints = (peaks[:-1] + peaks[1:]) // 2
            xdata = xdata[peak_midpoints[0]:peak_midpoints[-1]]
            ydata = ydata[peak_midpoints[0]:peak_midpoints[-1]]
            indices = np.where(ydata >= 3)[0] + peak_midpoints[0]
        else:
            indices = np.where(ydata >= 3)[0]

        return xdata[indices[-1]] - xdata[indices[0]]


    @staticmethod
    def oma(
        xdata: np.array, 
        ydata0: np.array, 
        ydata1: np.array,
        target_wavelength: float,
        normalised: bool=True,
        distance: float=100,
        threshold: float=8
    ) -> np.ndarray:
        """
        Calculate the optical modulation amplitude.

        Parameters:
            xdata (np.array): The x-axis data.
            ydata0 (np.array): The first y-axis data. Assume the data will be in dB form.
            ydata1 (np.array): The second y-axis data. Assume the data will be in dB form.
            target_wavelength (float): The target wavelength [m].
            distance (float): The distance between peaks.
            threshold (float): The threshold for the peaks.
            normalised (bool): Normalise the x-axis to resonance=0.

        Returns:
            float: The optical modulation amplitude.
        """
        # normalise before performing other operations
        # attempt to remove insertion loss difference
        peaks0 = find_peaks(ydata0, distance=distance)[0]
        peaks0 = peaks0[ydata0[peaks0] - min(ydata0) > threshold]
        
        peaks1 = find_peaks(ydata1, distance=distance)[0]
        peaks1 = peaks1[ydata1[peaks1] - min(ydata1) > threshold]

        target_idx = np.argmin(np.abs(xdata - target_wavelength))
        idx_max = peaks0[np.argmin(np.abs(peaks0 - target_idx))]
        idx_min = peaks1[np.argmin(np.abs(peaks1 - target_idx))]

        ydata0 = db2w(-(ydata0 - min(ydata0)))
        ydata1 = db2w(-(ydata1 - min(ydata1)))
        oma = np.absolute(ydata0 - ydata1)

        if normalised:
            min_idx0 = np.argmin(oma[idx_min:idx_max])
            xdata = xdata - xdata[idx_min + min_idx0]

        return oma
    
    @staticmethod
    def er(
        xdata: np.ndarray,
        ydata0: np.ndarray,
        ydata1: np.ndarray,
        target_wavelength: float,
        distance: float=100,
        threshold: float=8,
        normalised: bool=True,
    ) -> np.ndarray:
        """
        Calculate the extinction ratio.

        Parameters:
            xdata (np.array): The x-axis data.
            ydata0 (np.array): The first y-axis data. Assume the data will be in dB form.
            ydata1 (np.array): The second y-axis data. Assume the data will be in dB form.
            target_wavelength (float): The target wavelength [m].
            distance (float): The distance between peaks.
            threshold (float): The threshold for the peaks.
            normalised (bool): Normalise the x-axis to resonance=0.

        Returns:
            np.ndarray: Extinction ratio
        """
        # normalise before performing other operations
        peaks0 = find_peaks(ydata0, distance=distance)[0]
        peaks0 = peaks0[ydata0[peaks0] - min(ydata0) > threshold]
        
        peaks1 = find_peaks(ydata1, distance=distance)[0]
        peaks1 = peaks1[ydata1[peaks1] - min(ydata1) > threshold]

        target_idx = np.argmin(np.abs(xdata - target_wavelength))
        idx_max = peaks0[np.argmin(np.abs(peaks0 - target_idx))]
        idx_min = peaks1[np.argmin(np.abs(peaks1 - target_idx))]

        er = np.absolute(ydata0 - ydata1)

        if normalised:
            min_idx0 = np.argmin(er[idx_min:idx_max])
            xdata = xdata - xdata[idx_min + min_idx0]

        return er

    @staticmethod
    def operating_region(xdata: np.array, ydata: np.array, level: float):
        """
        Calculate the operating wavelength region of the resonator.
        Assume normalised data.

        Parameters:
            xdata (np.array): The x-axis data.
            ydata (np.array): OMA and ER data.
            level (float): The level of OMA to quantify as operating region.

        Returns:
            float: The operating wavelength of the resonator.
        """
        indices = np.where(ydata >= level)[0]
        
        if indices.size == 0:
            print("No operating region found.")
            return 0, 0
        mid = np.where(xdata == 0)[0][0]


        right_indices = indices[indices < mid]
        left_indices = indices[indices > mid]

        right_or = xdata[right_indices[0]] - xdata[right_indices[-1]] if right_indices.size > 0 else 0
        left_or = xdata[left_indices[0]] - xdata[left_indices[-1]] if left_indices.size > 0 else 0

        print(right_indices[0], right_indices[-1])
        # print(left_indices[0], left_indices[-1])

        return np.absolute(left_or), np.absolute(right_or)
    
    @staticmethod
    def fom_max(xdata: np.array, ydata: np.array, side: str):
        """
        FOM maximum assuming normalised data.
        """
        mid = np.where(xdata == 0)[0][0]
        if side == "left":
            return np.max(ydata[mid:])
        return np.max(ydata[:mid])

    def fwhm(self) -> float:
        """
        Calculate the full width at half maximum of the resonator.

        Returns:
            float: The full width at half maximum.
        """

        # Convert ydata from dB to linear ratio
        tol = 0.01
        ydata = db2w(-(self._ydata - min(self._ydata)))

        peaks = self._peaks_idx_for_averaging(3)
        peak_midpoints = (peaks[:-1] + peaks[1:]) // 2

        xdata = self._xdata[peak_midpoints[0]:peak_midpoints[-1]]
        ydata = ydata[peak_midpoints[0]:peak_midpoints[-1]]
        half_indices = np.where(np.isclose(ydata, 0.5, atol=tol))[0] + peak_midpoints[0]

        fwhm = xdata[half_indices[-1]] - xdata[half_indices[0]]

        return fwhm

 
    def qfactor(self) -> float:
        """
        Calculate the quality factor of the resonator.

        Returns:
            float: The quality factor of the resonator.
        """
        return self.closest_resonance()/self.fwhm()
        
    @staticmethod
    def total_capacitance(veff: np.array, vcap: np.array, eff: np.array, cap: np.array) -> np.array:
        """
        Calculate the total capacitance.

        Parameters:
            veff (np.array): Modulation efficiency voltage.
            vcap (np.array): Capacitance voltage.
            eff (np.array): Modulation efficiency.
            cap (np.array): Capacitance per length.

        Returns:
            np.array: The total capacitance.
        """
        voltages = np.concatenate((veff.flatten(), vcap.flatten()))
        voltages = np.linspace(min(voltages), max(voltages), 150)
        eff_func = interp1d(veff, eff, kind='linear', fill_value="extrapolate")
        cap_func = interp1d(vcap, cap, kind='linear', fill_value="extrapolate")

        total_cap = eff_func(voltages)*cap_func(voltages)/voltages

        return voltages, total_cap

    @staticmethod
    def find_phase_shift(wavelength: np.array, df: pd.DataFrame, target: float) -> float:
        """
        Find the phase shift from the transmission spectrum.

        Parameters:
            wavelength (np.array): The wavelength data.
            df (pd.DataFrame): The transmission spectrum data.
            target (float): The target wavelength.

        Returns:
            float: The phase shift.
        """
        res_shift = []
        fsr = 0
        for _, val in df.items():
            analysis = PAnalysis(wavelength, val, target, cutoff=5)
            res = analysis.closest_resonance()
            fsr += analysis.fsr()
            if res_shift == []:
                res_shift.append(0)
                res_at_0v = res
            else:
                res_shift.append(res - res_at_0v)

        # assume that the fsr
        fsr = fsr/len(df.keys())

        return 2*np.pi*np.array(res_shift)/fsr
        

    @staticmethod
    def get_modeff_from_dneff(wavelength: float, voltages: list, dneff: list):
        """
        This method calculate modulation efficiency based on the delta n

        Parameters
        ----------
        wavelength: float
            wavelength in m
        voltages: list
            list of voltages in V
        dneff: list
            list of delta neff
        """
        voltages = np.real(voltages)
        return voltages, (voltages*wavelength)/(2*np.real(dneff))
    
    @staticmethod
    def get_modeff_from_rshift(voltages: list, rshift: list, fsr: list, length: float):
        """
        This method calculate modulation efficiency based on the resonance shift

        Parameters
        ----------
        wavelength: float
            wavelength in m
        voltages: list
            list of voltages in V
        dneff: list
            list of delta neff
        """
        dphi_dv = np.absolute(2*np.pi*rshift/fsr)/voltages
        modeff = (length*np.pi)/dphi_dv
        return modeff
    
    def get_phase_shift_from_rshift(rshift: list, fsr: list):
        """
        This method calculate phase shift based on the resonance shift

        Parameters
        ----------
        rshift: list
            list of resonance shift
        fsr: list
            list of free spectral range
        """
        return 2*np.pi*rshift/fsr

    @staticmethod
    def get_loss(wavelength: float, voltages: list, dk: list):
        """
        This method calculate modulation efficiency based on the delta n

        Parameters
        ----------
        wavelength: float
            wavelength in m
        voltages: list
            list of voltages in V
        dneff: list
            list of delta neff
        """
        voltages = np.real(voltages)
        return voltages, (40*np.pi*dk*np.log10(np.e)/(wavelength))*1e-02 # [dB/cm]

    @staticmethod
    def get_modfrac(a: float, alpha: float, radius: float):
        """
        Get modulation fraction.

        Parameters
        ----------
        a: float
            Round trip loss
        alpha: float
            Loss per length [dB/cm]
        radius: float
            Radius of the ring [um]
        """
        length = -20*np.log10(a)/(alpha*10**2)
        return length/(2*np.pi*radius)

    def get_loss_from_a(a: list, length: float):
        """
        Get loss from a.

        Parameters
        ----------
        a: list
            List of round trip loss

        a**2 = exp(-alpha*length)
        """
        a_db = 20*np.log10(np.array(a)) # converting a to db
        alpha_db = -a_db/(10*np.log10(np.e)*length)
        return alpha_db


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

    peaks = analysis.peaks
    plt.plot(wavelength*1e+09, ydata, label="data")
    plt.plot(wavelength[peaks]*1e+09, ydata[peaks], "x", label="peaks")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Loss [dB]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    