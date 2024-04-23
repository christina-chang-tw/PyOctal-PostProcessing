"""
fitting.py

Fitting the transmission spectrum
"""
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from postprocessing.analysis import PAnalysis
from matplotlib import pyplot as plt


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


def ring_transmission(xdata: np.ndarray, ydata: np.ndarray, peaks: np.ndarray, 
                     fsr: float, target_wavelength: float, fixed: float = None) -> tuple:
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
    """
    xlow, xhigh, ridx = find_spectrum_indices(xdata, peaks, target_wavelength)

    xdata = xdata[xlow:xhigh]
    ydata = ydata[xlow:xhigh]

    lambda_r = xdata[ridx]

    bounds = ((0.5, 0.5, 0), (1, 1, 1))
    initial_guess = 0.6
    wrapper_func = lambda x, var, p: ring_resonator_model(x, lambda_r, fsr, fixed, var)*p

    if fixed is None:
        # Fit the model with both amplitude and reflectivity as free parameters
        initial_guess = (initial_guess, initial_guess, 0.6)
        wrapper_func = lambda x, a, t, p: ring_resonator_model(x, lambda_r, fsr, a, t)*p

    params, *_ = curve_fit(wrapper_func, xdata, ydata, bounds=bounds, p0=initial_guess)

    return params


def find_spectrum_indices(xdata: np.array, peaks: float, wavelength: float) -> list[int, int, int]:
    """
    Find the indices of the spectrum that correspond to the resonance wavelength.

    Parameters:
        xdata (np.array): The x-axis data.
        peaks (list): An array of peak positions.
        wavelength (float): The wavelength of interest.

    Returns:
        tuple: A tuple containing (left, right, resonance) indices.
    """
    # Find the closest peak to the target wavelength
    target_idx = np.argmin(np.abs(peaks - np.argmin(np.abs(xdata - wavelength))))

    # Extract a small range of data around the peak
    indices_of_interest = peaks[np.arange(target_idx - 1, target_idx + 2)]

    ilow, ihigh = (indices_of_interest[:-1] + indices_of_interest[1:]) // 2

    return ilow, ihigh, peaks[target_idx]


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


def ramzi_transmission(ring: dict, target_wavelength: float, mzi: Union[dict,list[dict]], balance: bool=True) -> np.ndarray:
    i0 = 1

    a_r = 1
    phi_r = 1

    if balance is True:
        delta_phi = mzi["delta_phi"]
        mzi_amp_atten = np.exp(-mzi["alpha"]*mzi["L"])
        iout = (i0 * mzi_amp_atten**2)*[1 + a_r**2 + 2 * a_r*np.cos(delta_phi + phi_r)]/4
    
    else:
        if len(mzi) == 2:
            raise ValueError("Two MZIs are required for unbalanced operation.")
        
        mzi1 = mzi[0]
        mzi2 = mzi[1]

        mzi1_amp_atten = np.exp(-mzi1["alpha"]*mzi1["L"])
        mzi2_amp_atten = np.exp(-mzi2["alpha"]*mzi2["L"])

        phi1 = mzi1["beta"]*mzi1["L"]
        phi2 = mzi2["beta"]*mzi2["L"]

        iout = (i0*(mzi1_amp_atten**2*a_r**2 + mzi2_amp_atten**2 + 2*mzi1_amp_atten*mzi2_amp_atten*a_r*np.cos(phi1 + phi_r - phi2)))/4



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

    voltages = range(0, 4)
    filenames = [f"2024-2-27-ring/Radius1_t17_{v}v.csv" for v in voltages]
    fig, axes = plt.subplots(1, 4)
    for idx, filename in enumerate(filenames):

        df = pd.read_csv(filename)
        target_wavelength = 1550e-09
        wavelength = df["Wavelength"]
        ydata = df["Loss [dB]"] # this ydata should be positive.

        analysis = PAnalysis(wavelength, ydata, target_wavelength, cutoff=10, distance=100)
        fsr = analysis.fsr()
        print(analysis.peaks)

        ydata = 10**(-ydata/10)

        xlow, xhigh, _ = find_spectrum_indices(wavelength, analysis.peaks, target_wavelength)


        a, t, p = ring_transmission(df["Wavelength"], ydata, peaks=analysis.peaks,
                                fsr=fsr, target_wavelength=target_wavelength)
        
        print(f"Voltage: {voltages[idx]}, Amplitude: {a}, Transmission: {t}, Scaling by loss: {p}")

        target_idx = np.argmin(np.abs(analysis.peaks - np.argmin(np.abs(wavelength - target_wavelength))))
        lambda_r = wavelength[analysis.peaks[target_idx]]
        ydata_fit = ring_resonator_model(df["Wavelength"], lambda_r, fsr, a, t)*p

        axes[idx].plot(wavelength[xlow:xhigh], ydata[xlow:xhigh], label="original")
        axes[idx].plot(wavelength[xlow:xhigh], ydata_fit[xlow:xhigh], label="fitted")
        axes[idx].legend()

    fig.tight_layout()
    
    plt.show()
 


if __name__ == "__main__":
    main()
