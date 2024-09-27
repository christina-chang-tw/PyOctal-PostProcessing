"""
fitting.py

Fitting the transmission spectrum
"""
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from postprocessing.analysis import PAnalysis


def rearrange_ring_data(x, y, ind: str = "a"):
    """
    Rearrange the data from the ring resonator and give back the appropriate self-coupling (t)
    and ring energy field (a).
    i.e.
        If ind is "a", then the data is rearranged to make sure 'a' remains constant.
        If ind is "t", then the data is rearranged to give the ring energy field.

    Parameters:
        x (np.ndarray): The x-axis data.
        y (np.ndarray): The y-axis data.
        ind (str, optional): The independent variable.

    Returns:
        tuple: A tuple containing the x-axis and y-axis data.
    """
    inds = []
    deps = []
    for xdata, ydata in zip(x, y):
        inds.append

    return xdata, ydata

def ring_resonator_model(xdata: np.ndarray, lambda_r: float,
                    fsr: float, a: float, t: float) -> np.ndarray:
    """
    Resonator model function.

    Parameters:
        xdata (np.ndarray): The x-axis data.
        lambda_r (float): The resonance wavelength.
        fsr (float): The free spectral range of the resonator.
        a (float): The amplitude parameter.
        r (float): The reflectivity parameter.

    Returns:
        np.ndarray: The calculated transmission values.
    """
    phi = 2*np.pi*(xdata - lambda_r) / fsr
    transmission = (a**2 - 2*a*t*np.cos(phi) + t**2) / (1 - 2*a*t*np.cos(phi) + (a*t)**2)

    return transmission

def ramzi_output_intensity(xdata, lambda_r, fsr, dphi, a, t):
    phi = 2*np.pi*(xdata - lambda_r) / fsr
    ring_ph = np.pi + phi + np.arctan2((t*a*np.sin(phi)),(1-t*a*np.cos(phi))) + np.arctan2((t*np.sin(phi)),(a-t*np.cos(phi)))

    field = (t-a*np.exp(1j*phi))/(1-t*a*np.exp(1j*phi))
    transmission = (a**2 - 2*a*t*np.cos(phi) + t**2) / (1 - 2*a*t*np.cos(phi) + (a*t)**2)

    return (transmission + 1 + 2*np.abs(field)*np.cos(dphi + ring_ph))/4

def curve_fitting_for_ramzi(xdata: np.array, ydata: np.array,
                            lambda_r: float, fsr: float, dphi: float, fixed_param: float = None) -> tuple:
    """
    Fit the output intensity spectrum of a Ramzi interferometer.

    Parameters:
        xdata (np.array): The x-axis data.
        ydata (np.array): The y-axis data.
        lambda_r (float): The resonance wavelength.
        fsr (float): The free spectral range of the interferometer.
        dphi (float): The phase difference between the two arms of the interferometer.
        fixed_param (float, optional): If provided, the amplitude parameter is fixed to this value.

    Returns:
        tuple: A tuple containing the fitted parameters (amplitude, transmission, scaling factor).
    """
    def ramzi_model_wrapper(x, a, t, p):
        return ramzi_output_intensity(x, lambda_r, fsr, dphi, a, t)*p

    def ramzi_model_wrapper_fixed(x, var, p):
        return ramzi_output_intensity(x, lambda_r, fsr, dphi, fixed_param, var)*p
    
    if fixed_param is None:
        bounds = ((0.5, 0.5, 0), (0.98, 1, 1))
        initial_guess = (0.85, 0.9, 0.1)
        popt, _ = curve_fit(ramzi_model_wrapper, xdata, ydata, bounds=bounds, p0=initial_guess)
    else:
        bounds = ((0.5, 0), (1, 1))
        initial_guess = (0.85, 0.1)
        popt, _ = curve_fit(ramzi_model_wrapper_fixed, xdata, ydata, bounds=bounds, p0=initial_guess)

    return popt

def curve_fitting_for_ring(xdata: np.ndarray, ydata: np.ndarray, lambda_r: float, 
                     fsr: float, fixed_param: float = None, p_fixed: float=None, bounds: list=None) -> tuple:
    """
    Fit the transmission spectrum of the resonator.

    Parameters:
        xdata (np.ndarray): The x-axis data.
        ydata (np.ndarray): The y-axis data in linear form. Please convert db to linear first.
        peaks (np.ndarray): An array of peak positions.
        fsr (float): The free spectral range of the resonator.
        target_wavelength (float): The wavelength of interest.
        fixed_amplitude (float, optional): If provided, 
            the amplitude parameter is fixed to this value.

    Returns:
        tuple or float: A tuple containing the fitted parameter(s) 
            (loss, cross-coupling coefficient)

        ** Change the boundary conditions for the fitting parameters can help a lot! ** 
    """
    def ring_resonator_model_wrapper(x, a, t, p):
        return ring_resonator_model(x, lambda_r, fsr, a, t) * p

    def ring_resonator_model_wrapper_fixed(x, var, p):
        return ring_resonator_model(x, lambda_r, fsr, fixed_param, var) * p
    
    def ring_resonator_model_wrapper_fixed_p(x, a, t):
        return ring_resonator_model(x, lambda_r, fsr, a, t) * p_fixed

    if fixed_param is None:
        # Fit the model with both amplitude and reflectivity as free parameters
        bounds = ((0.8, 0.8, 0), (0.99, 1, 0.5))
        initial_guess = (0.9, 0.93, 0.1)
        params, *_ = curve_fit(ring_resonator_model_wrapper, xdata, ydata, bounds=bounds, p0=initial_guess)
    elif p_fixed is not None:
        # Fix the scaling factor and fit the model
        bounds = ((0.9, 0.9), (0.99, 0.99)) if bounds is None else bounds
        initial_guess = (0.93, 0.94)
        params, *_ = curve_fit(ring_resonator_model_wrapper_fixed_p, xdata, ydata, bounds=bounds, p0=initial_guess)
    else:
        # Fix the transmission coefficient parameter and fit the model
        bounds = ((0.8, 0), (0.99, 0.5)) if bounds is None else bounds
        initial_guess = (0.932, 0.17)
        params, *_ = curve_fit(ring_resonator_model_wrapper_fixed, xdata, ydata, bounds=bounds, p0=initial_guess)

    return params


def ring_fit_interface(
        df: pd.DataFrame, 
        target_wavelength: float=1550E-09, 
        fixed: float=None, 
    ) -> Tuple:
    wavelength = df["Wavelength"]
    ydata = df["Loss [dB]"]
    analysis = PAnalysis(wavelength, ydata, target_wavelength, cutoff=2, distance=100)
    fsr = analysis.fsr()

    xlow, xhigh = analysis.get_range_idx(range=1E-09)    
    lambda_r = analysis.closest_resonance()

    wavelength = wavelength[xlow:xhigh]
    ydata = ydata[xlow:xhigh]
    ydata_linear = 10**(-ydata/10)

    return wavelength, ydata, ydata_linear, curve_fitting_for_ring(wavelength, ydata_linear, lambda_r, fsr, fixed)