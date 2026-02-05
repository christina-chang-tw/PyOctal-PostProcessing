"""
analysis.py

Perform photonics simulation/experimental data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from scipy.signal import find_peaks

class PAnalysis:
    """
    This is for performing analysis based on the transmission spectrum.

    Parameters:
        xdata (np.ndarray): The x-axis data (m).
        ydata (np.ndarray): The y-axis data (dB).
        wavelength (float): The wavelength of interest in m.  
    """
    def __init__(self, xdata: np.array, ydata: np.array, wavelength: float, cutoff: float=20, distance: int=100):
        self._xdata = xdata
        self._ydata = np.absolute(ydata)
        self.target_wavelength = wavelength
        self.target_wavelength_idx = np.argmin(np.abs(self._xdata - self.target_wavelength))
        self._peaks_idx = self.get_resonance_indices(cutoff=cutoff, distance=distance)
        self._peaks = self._xdata[self._peaks_idx]
      

    def sanity_check(self, xlim: list=None, ylim: list=None):
        """
        Check your human brain sanity.
        """
        plt.clf()
        plt.plot(self.xdata*1E+09, self.ydata)
        plt.scatter(self.xdata[self._peaks_idx]*1E+09, self.ydata[self._peaks_idx], marker="x")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmission (dB)")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()

    @property
    def xdata(self):
        """ 
        Get the x-axis data.
        """
        return self._xdata
    
    @property
    def ydata(self):
        """ 
        Get the y-axis data.
        """
        return self._ydata

    @property
    def peaks_idx(self):
        return self._peaks_idx

    @property
    def peaks(self):
        """ 
        Get the peaks in the spectrum.
        """
        return self._peaks

    @property
    def true_resonance_idx(self):
        """ 
        Get the index that is closest to the target wavelength in the peak.
        """
        return self.peaks_idx[np.argmin(np.abs(self.peaks_idx - self.target_wavelength_idx))]
    
    @property
    def true_offres_idx(self):
        """ 
        Get the power maximum index that is closest to the target wavelength in the original data. 
        """
        if self.target_resonance_idx < self.target_wavelength_idx:
            return (self._peaks[self.true_resonance_idx] + self._peaks[self.true_resonance_idx + 1]) // 2
        return (self._peaks[self.true_resonance_idx] + self._peaks[self.true_resonance_idx - 1]) // 2
    
    @property
    def true_resonance_wavelength(self):
        """ 
        Get the resonance wavelength that is closest to the target wavelength in the orriginal data. 
        """
        return self.peaks[np.argmin(np.abs(self.peaks - self.target_wavelength))]

    def get_range_idx(self, xrange: int=1E-09):
        """
        Get the range of the data closest to the resonance peak that is closest to the target
        wavelength. Return two indices that correspond to the range which encloses one peak.
        idx_low corresponds to lower frequency/higher wavelength.

        Parameters:
            range (int): The range of the data to be considered. int=1 means 1nm range.
        """
        res = self.true_resonance_wavelength
        idx_min = np.argmin(np.abs(self.xdata - (res - xrange/2)))
        idx_max = np.argmin(np.abs(self.xdata - (res + xrange/2)))

        return idx_min, idx_max

    def centering(self, idx: int) -> np.ndarray:
        """
        Center the x-axis data with respect to the resonance frequency.

        Parameters:
            idx (int): The index of the resonance frequency.

        Returns:
            np.ndarray: The centered x-axis data.
        """
        lmax_idx = np.argmax(self.ydata[:idx])
        rmax_idx = np.argmax(self.ydata[idx:]) + idx
        min_idx0 = np.argmin(self.ydata[lmax_idx:rmax_idx])
        xdata = self.xdata - self.xdata[lmax_idx + min_idx0]
        return xdata

    def modeff_phase(self, voltage: np.array, phase: np.array, length: float) -> np.array:
        """
        Modulation efficiency Vm.
        """
        return length * voltage * np.pi / phase


    def get_resonance_indices(self, cutoff: float, distance: int) -> list:
        """
        Find the peaks in the spectrum.

        Parameters:
            cutoff (float): The cutoff value for the peaks.
            distance (int): The minimum distance between peaks.

        Returns:
            list: The indices of the peaks.
        """
        peaks_idx, _ = find_peaks(self.ydata, distance=distance)

        # perform another filtering
        peaks_idx = peaks_idx[self.ydata[peaks_idx] - min(self.ydata) > cutoff]
        return peaks_idx
   
    def _peaks_idx_for_averaging(self, num: int) -> list:
        """
        Find the peaks for averaging.

        Parameters:
            num (int): The number of peaks.

        Returns:
            list: The indices of the peaks for averaging.
        """
        if num >= len(self.peaks_idx):
            return self.peaks_idx

        index = np.argmin(np.abs(self.peaks_idx - self.true_resonance_idx))

        peaks_for_avg = self.peaks_idx[index-num//2:index+1+num//2]

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
            fsr += self.xdata[peaks_idx[i]] - self.xdata[peaks_idx[i-1]]
        
        return fsr / (len(peaks_idx)-1)
    
    def linewidth(self) -> float:
        """
        Calculate the linewidth of the resonator.
        Must provide linear scale ydata

        Returns:
            float: The linewidth of the resonator.
        """
        xdata = self.xdata
        ydata = self.ydata

        if len(self._peaks_idx) > 1:
            peaks_idx = self._peaks_idx_for_averaging(3)
            peak_midpoints = (peaks_idx[:-1] + peaks_idx[1:]) // 2
            ydata = ydata[peak_midpoints[0]:peak_midpoints[-1]]
            indices = np.where(ydata >= 3)[0] + peak_midpoints[0]
        else:
            indices = np.where(ydata >= 3)[0]

        return np.absolute(self.xdata[indices[-1]] - self.xdata[indices[0]])


    @staticmethod
    def fom(
        xdata: np.array, 
        ydata0: np.array, 
        ydata1: np.array,
        target_wavelength: float,
        fom_type: str="OMA",
        peak_distance: int=100,
        peak_depth: int=8,
    ) -> tuple:
        """
        Calculate the figure of merit.

        Parameters:
            xdata (np.array): The x-axis data.
            ydata0 (np.array): The first y-axis data. Assume the data will be in dB form.
            ydata1 (np.array): The second y-axis data. Assume the data will be in dB form.

        Returns:
            float: The optical modulation amplitude.
        """
        # normalise before performing other operations
        peaks0 = find_peaks(ydata0, distance=peak_distance)[0]
        peaks0 = peaks0[ydata0[peaks0] - min(ydata0) > peak_depth]
        
        peaks1 = find_peaks(ydata1, distance=peak_distance)[0]
        peaks1 = peaks1[ydata1[peaks1] - min(ydata1) > peak_depth]

        target_idx = np.argmin(np.abs(xdata - target_wavelength))
        idx_max = peaks0[np.argmin(np.abs(peaks0 - target_idx))]
        idx_min = peaks1[np.argmin(np.abs(peaks1 - target_idx))]

        if fom_type == "ER":
            return np.absolute(ydata0 - ydata1)

        ydata0 = 10**(-(ydata0 - min(ydata0)) / 10)
        ydata1 = 10**(-(ydata1 - min(ydata1))/ 10)

        if fom_type == "OMA":
            return np.absolute(ydata0 - ydata1)

        if fom_type == "TP":
            return -10 * np.log10(np.absolute(ydata0 - ydata1) / 2)


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
        ydata = 10**(-(self.ydata - min(self.ydata)) / 10)

        peaks_idx = self._peaks_idx_for_averaging(3)
        peak_midpoints = (peaks_idx[:-1] + peaks_idx[1:]) // 2

        ydata = ydata[peak_midpoints[0]:peak_midpoints[-1]]
        half_indices = np.where(np.isclose(ydata, 0.5, atol=tol))[0] + peak_midpoints[0]

        fwhm = np.absolute(self.xdata[half_indices[-1]] - self.xdata[half_indices[0]])

        return fwhm

 
    def qfactor(self) -> float:
        """
        Calculate the quality factor of the resonator.

        Returns:
            float: The quality factor of the resonator.
        """
        return self.true_resonance_wavelength/self.fwhm()

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
    def er(self) -> float:
        """
        Only allow one trough and find the extinction ratio
        """
        return max(self.ydata) - min(self.ydata)
        

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
