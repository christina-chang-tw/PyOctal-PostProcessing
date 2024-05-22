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
        xdata (np.ndarray): The x-axis data. [dB]
        ydata (np.ndarray): The y-axis data. [dB] 
        wavelength (float): The wavelength of interest.  
    """
    def __init__(self, xdata: np.array, ydata: np.array, wavelength: float, cutoff: float=20, distance: int=100):
        self.xdata = xdata
        self.ydata = np.absolute(ydata)
        self.wavelength = wavelength
        self.wavelength_idx = np.argmin(np.abs(self.xdata - self.wavelength))
        self._peaks = self.resonances(cutoff=cutoff, distance=distance)

    @property
    def peaks(self):
        return self._peaks

    @property
    def target_peak_idx(self):
        """ 
        Get the index that is closest to the target wavelength in the peak. """
        return np.argmin(np.abs(self._peaks - self.wavelength_idx))
    
    @property
    def target_offres_idx(self):
        """ 
        Get the power maximum index that is closest to the target wavelength in the orriginal data. 
        """
        if self.target_resonance_idx < self.wavelength_idx:
            return (self._peaks[self.target_peak_idx] + self._peaks[self.target_peak_idx + 1]) // 2

        return (self._peaks[self.target_peak_idx] + self._peaks[self.target_peak_idx - 1]) // 2

    @property
    def target_resonance_idx(self):
        """ 
        Get the resonance index that is closest to the target wavelength in the orriginal data. 
        """
        return self._peaks[self.target_peak_idx]
    
    @property
    def target_resonance_wavelength(self):
        """ 
        Get the resonance wavelength that is closest to the target wavelength in the orriginal data. 
        """
        return self.xdata[self.target_resonance_idx]


    def resonance_freq(self) -> float:
        """
        Calculate the resonance frequency of the resonator.

        Returns:
            float: The resonance frequency closest to the target wavelength.
        """
        target_idx = np.argmin(np.abs(self._peaks - self.wavelength_idx))
        return self.xdata[self._peaks[target_idx]]


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
        peaks, _ = find_peaks(self.ydata, distance=distance)

        # perform another filtering
        peaks = peaks[self.ydata[peaks] - min(self.ydata) > cutoff]

        return peaks
   
    def _peaks_idx_for_averaging(self, num: int) -> list:
        """
        Find the peaks for averaging.

        Parameters:
            num (int): The number of peaks.

        Returns:
            list: The indices of the peaks for averaging.
        """
        target_idx = np.argmin(np.abs(self.peaks - self.wavelength_idx))
        peaks_for_avg = self.peaks[target_idx-num//2:target_idx+1+num//2]

        return peaks_for_avg

    def fsr(self) -> float:
        """
        Calculate the free spectral range of the resonator.


        Returns:
            float: The free spectral range of the resonator.
        """
        # find the closest peaks to the resonance wavelength
        peaks_idx = self._peaks_idx_for_averaging(5)

        # takes averaging
        fsr = 0
        for i in range(1, len(peaks_idx)):
            fsr += self.xdata[peaks_idx[i]] - self.xdata[peaks_idx[i-1]]
        
        return fsr/len(peaks_idx)
    
    def linewidth(self) -> float:
        """
        Calculate the linewidth of the resonator.
        Must provide linear scale ydata

        Returns:
            float: The linewidth of the resonator.
        """
        peaks = self._peaks
        xdata = self.xdata
        ydata = self.ydata

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

        return left_or, right_or
    
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

        peaks = self._peaks_idx_for_averaging(3)
        peak_midpoints = (peaks[:-1] + peaks[1:]) // 2

        xdata = self.xdata[peak_midpoints[0]:peak_midpoints[-1]]
        ydata = ydata[peak_midpoints[0]:peak_midpoints[-1]]
        half_indices = np.where(np.isclose(ydata, 0.5, atol=tol))[0]

        fwhm = xdata[half_indices[-1]] - xdata[half_indices[0]]

        return fwhm

 
    def qfactor(self) -> float:
        """
        Calculate the quality factor of the resonator.

        Returns:
            float: The quality factor of the resonator.
        """
        return self.resonance_freq()/self.fwhm()
    
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
    def get_modeff(wavelength: float, voltages: list, dneff: list):
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
    