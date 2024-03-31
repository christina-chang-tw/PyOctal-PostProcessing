"""
fitting.py

Fitting the transmission spectrum
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from postprocessing.analysis import PAnalysis

def resonator_model(xdata: np.ndarray, lambda_r: float,
                    fsr: float, a: float, r: float) -> np.ndarray:
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
    transmission = (a**2 - 2*a*r*np.cos(phi) + r**2) / (1 - 2*a*r*np.cos(phi) + (a*r)**2)

    return transmission


def fit_transmission(xdata: np.ndarray, ydata: np.ndarray, peaks: np.ndarray, 
                     fsr: float, target_wavelength: float, fixed: float = None) -> tuple:
    """
    Fit the transmission spectrum of the resonator.

    Parameters:
        xdata (np.ndarray): The x-axis data.
        ydata (np.ndarray): The y-axis data in dB form.
        peaks (np.ndarray): An array of peak positions.
        fsr (float): The free spectral range of the resonator.
        target_wavelength (float): The wavelength of interest.
        fixed_amplitude (float, optional): If provided, 
            the amplitude parameter is fixed to this value.

    Returns:
        tuple or float: A tuple containing the fitted parameter(s) 
            (loss, cross-coupling coefficient) 
    """
    # Find the closest peak to the target wavelength
    target_idx = np.argmin(np.abs(peaks - np.argmin(np.abs(xdata - target_wavelength))))
    lambda_r = xdata[peaks[target_idx]]

    # Extract a small range of data around the peak
    peak_indices = np.arange(target_idx - 1, target_idx + 2)
    peak_wavelengths = peaks[peak_indices]
    peak_midpoints = (peak_wavelengths[:-1] + peak_wavelengths[1:]) // 2
    xlow, xhigh = peak_midpoints[0], peak_midpoints[1]
    xdata = xdata[xlow:xhigh]
    ydata = ydata[xlow:xhigh]

    # Convert ydata from dB to linear ratio
    ydata = 10**(-(ydata - min(ydata)) / 10)

    bounds = (0.5, 1)
    initial_guess = 0.6
    wrapper_func = lambda x, var: resonator_model(x, lambda_r, fsr, fixed, var)

    if fixed is None:
        # Fit the model with both amplitude and reflectivity as free parameters
        initial_guess = (initial_guess, initial_guess)
        wrapper_func = lambda x, a, r: resonator_model(x, lambda_r, fsr, a, r)

    params, *_ = curve_fit(wrapper_func, xdata, ydata, bounds=bounds, p0=initial_guess)

    return params

def imag_omega_r(ne: float, nh: float, lambda_r: float) -> float:
    """
    Calculate the imaginary part of the resonant frequency.

    Parameters:
        ne (float): The electron density in cm-3.
        nh (float): The hole density in cm-3.
        lambda_r (float): The resonance wavelength in cm.

    Returns:
        float: The imaginary part of the resonant frequency.
    """
    weight = nh/ne

    delta_n = -8.8E-22*weight*ne-8.5E-18*nh**0.8
    delta_alpha_m = 8.5E-18*weight*ne+6.0E-18*nh

    imag_delta_omega_r = - (lambda_r/(4*np.pi))*((delta_alpha_m)/delta_n)

    return imag_delta_omega_r


# def peaking_amplifier_model():
#     Av = 1
#     R1 = 10
#     wavelength = np.arange(1.5e-6, 1.6e-6, 1e-9)
#     c = 299792458
#     omega = 2*np.pi*c/wavelength

#     R2 = 50
#     C = 1e-12
#     L = 1e-9
#     R2_bar = 1/(Av+1)*R2/R1
#     L_bar = 1/(Av+1)*L/R1
#     C_bar = C*R1

#     gain = - Av*(R2_bar-1j*L_bar*omega)/(1+R2_bar-1j*L_bar*omega-1j*C_bar*R2_bar*omega-C_bar*L_bar*omega**2)

#     return gain

# def fit_s21():
#     """
#     Fit the transmission spectrum of the resonator.
#     """
#     Ein0 = 1
#     a = 1
#     t = 0
#     N_e = 1 # electron density in cm-3
#     N_h = 2 # Hole density in cm-3

    # mu = np.sqrt(kappa*vg/L)

    # Ein = Ein0*np.exp(1j*omega_0*t)
    # a = a0*np.exp(1j*omega_0*t)
    # Eout = Ein + 1j*a

    # da_dt = (-1j*omega_r - 1/tau)*a + 1j*mu*Ein
    # delta_n = -8.8*10**-22*N_e - 8.5*10**-18*N_h**0.8
    # delta_a = 8.5*10**-18*N_e + 6.0*10**-18*N_h

    # omode = partial_neff/partial_n # overlap of the waveguide mode with the region
    # real_delta_omega_r = -(omode*delta_n/n_g)*omega_r

    # delta_photon_lifetime = omode*delta_alpha/alpha


    

    # omega_r = omega_r + delta_omega_r*np.sin(omega_m*t)

    # s21 = mu*np.real((((delta_omega_r*a*np.conj(Ein+1j*mu*a))/(1/tau_a-1j*omega_m-(1j*omega_0-1j*omega_r))+(np.conj(delta_omega_r)*np.conj(a)*(Ein+1j*mu*a))/(1/tau_a-1j*omega_m+(1j*omega_0-1j*omega_r)))*np.exp(-1j*omega_m*t)))



def main():
    filename = "output/Radius1_t17_3v.csv"

    df = pd.read_csv(filename)
    target_wavelength = 1550e-09
    wavelength = df["Wavelength"]
    ydata = df["Loss [dB]"]

    analysis = PAnalysis(wavelength, ydata, target_wavelength, cutoff=10, distance=100)
    fsr = analysis.fsr()

    a, r = fit_transmission(df["Wavelength"], df["Loss [dB]"], peaks=analysis.peaks, 
                            fsr=fsr, target_wavelength=target_wavelength)
    print(a, r)

    target_idx = np.argmin(np.abs(analysis.peaks - np.argmin(np.abs(wavelength - target_wavelength))))
    lambda_r = wavelength[analysis.peaks[target_idx]]
    ydata = resonator_model(df["Wavelength"], lambda_r, fsr, a, r)
 


if __name__ == "__main__":
    main()
