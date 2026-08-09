"""
fom.py

Figure-of-merit calculations (extinction ratio, optical modulation
amplitude, transmitted power) derived from a pair of transmission spectra.
"""
import numpy as np


def fom(
    ydata0: np.array,
    ydata1: np.array,
    fom_type: str = "OMA",
    normalised: bool = True,
) -> np.ndarray:
    """
    Calculate the figure of merit between two biasing conditions of a modulator.

    Parameters:
        ydata0 (np.array): The first y-axis data. Assume the data will be in dB form.
        ydata1 (np.array): The second y-axis data. Assume the data will be in dB form.
        fom_type (str): One of "ER" (extinction ratio, dB), "OMA" (optical
            modulation amplitude, linear), "TP" (transmitted power, dB).
        normalised (bool): Normalise each spectrum to its own peak transmission
            before computing OMA/TP.

    Returns:
        np.ndarray: The figure of merit spectrum.
    """
    if fom_type == "ER":
        return np.absolute(ydata1 - ydata0)

    power0 = 10**(-(ydata0 - min(ydata0)) / 10) if normalised else 10**(-ydata0 / 10)
    power1 = 10**(-(ydata1 - min(ydata1)) / 10) if normalised else 10**(-ydata1 / 10)
    oma = np.absolute(power0 - power1)

    if fom_type == "OMA":
        return oma
    if fom_type == "TP":
        return -10 * np.log10(oma / 2)

    raise ValueError(f"Unknown fom_type: {fom_type!r}. Expected 'ER', 'OMA', or 'TP'.")


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


def fom_max(xdata: np.array, ydata: np.array, side: str):
    """
    FOM maximum assuming normalised data.
    """
    mid = np.where(xdata == 0)[0][0]
    if side == "left":
        return np.max(ydata[mid:])
    return np.max(ydata[:mid])
