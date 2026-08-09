"""
modulator.py

Device-level modulation efficiency, capacitance, and loss calculations for
electro-optic modulators.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from postprocessing.analysis import PAnalysis


def modeff_phase(voltage: np.ndarray, phase: np.ndarray, length: float) -> np.ndarray:
    """
    Modulation efficiency Vm.

    Parameters:
        voltage (np.ndarray): The applied voltages.
        phase (np.ndarray): The phase of the wavelength.
        length (float): The length of the device.

    Return:
        np.ndarray: The modulation efficiency
    """
    return length * voltage * np.pi / phase


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


def find_phase_shift(wavelength: np.ndarray, df: pd.DataFrame, target: float) -> np.ndarray:
    """
    Find the phase shift from the transmission spectrum.

    Parameters:
        wavelength (np.array): The wavelength data.
        df (pd.DataFrame): The transmission spectrum data.
        target (float): The target wavelength.

    Returns:
        np.ndarray: The phase shift.
    """
    res_shift = []
    fsr = 0
    for _, val in df.items():
        analysis = PAnalysis(wavelength, val, target, cutoff=5)
        res = analysis.resonance_wavelength
        fsr += analysis.fsr()
        if res_shift == []:
            res_shift.append(0)
            res_at_0v = res
        else:
            res_shift.append(res - res_at_0v)

    # assume that the fsr
    fsr = fsr/len(df.keys())

    return 2*np.pi*np.array(res_shift)/fsr


def get_modeff_from_dneff(wavelength: float, voltages: np.ndarray, dneff: np.ndarray) -> np.ndarray:
    """
    This method calculate modulation efficiency based on the delta n

    Parameters
    ----------
    wavelength: float
        wavelength in m
    voltages: np.ndarray
        An array of voltages
    dneff: np.ndarray
        An array of change in effective index

    Returns
        np.ndarray: The modulation efficiency fomr change in effective index.
    """
    voltages = np.real(voltages)
    return voltages, (voltages*wavelength)/(2*np.real(dneff))


def get_modeff_from_rshift(voltages: np.ndarray, rshift: np.ndarray, fsr: np.ndarray, length: float) -> np.ndarray:
    """
    This method calculate modulation efficiency based on the resonance shift

    Parameters
    ----------
    voltages: np.ndarray
        An array of voltages.
    rshift: np.ndarray
        The resonance shift.
    fsr: np.ndarray
        FSR values for each voltages.
    length: np.ndarray
        The length of the device.

    Returns
    -------
    np.ndarray:
        The modulation efficiency corresponding to different voltages
    """
    dphi_dv = np.absolute(2*np.pi*rshift/fsr)/voltages
    modeff = (length*np.pi)/dphi_dv
    return modeff


def get_phase_shift_from_rshift(rshift: np.ndarray, fsr: np.ndarray) -> np.ndarray:
    """
    This method calculate phase shift based on the resonance shift

    Parameters
    ----------
    rshift: np.ndarray
        An array of resonance shift.
    fsr: np.ndarray
        An array of free spectral range (FSR).
    """
    return 2*np.pi*rshift/fsr


def get_loss(wavelength: float, voltages: np.ndarray, dk: np.ndarray) -> np.ndarray:
    """
    This method calculate modulation efficiency based on the delta n

    Parameters
    ----------
    wavelength: float
        wavelength in m
    voltages: np.ndarray
        An array of voltages
    dk: np.ndarray
        An array of the imaginary part of the effective index

    Returns
    -------
    np.ndarray:
        The loss in dB/cm
    """
    voltages = np.real(voltages)
    return voltages, (40*np.pi*dk*np.log10(np.e)/(wavelength))*1e-02  # [dB/cm]


def get_modfrac(a: float, alpha: float, radius: float) -> float:
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

    Returns
    -------
    float:
        Modulation fraction
    """
    length = -20*np.log10(a)/(alpha*10**2)
    return length/(2*np.pi*radius)


def get_loss_from_a(a: np.ndarray, length: float) -> float:
    """
    Get loss from a.

    Parameters
    ----------
    a: np.ndarray
        List of round trip loss

    a**2 = exp(-alpha*length)
    """
    a_db = 20*np.log10(np.array(a))  # converting a to db
    alpha_db = -a_db/(10*np.log10(np.e)*length)
    return alpha_db
