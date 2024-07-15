"""
fitting.py

Fitting the transmission spectrum
"""
from typing import Union, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from postprocessing.analysis import PAnalysis
from matplotlib import pyplot as plt

from postprocessing.utils.formatter import Publication
from postprocessing.parser import Parser

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
                     fsr: float, fixed_param: float = None, bounds: list=None) -> tuple:
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

    if fixed_param is None:
        # Fit the model with both amplitude and reflectivity as free parameters
        bounds = ((0.5, 0.5, 0), (0.98, 1, 1))
        initial_guess = (0.7, 0.9, 0.2)
        params, *_ = curve_fit(ring_resonator_model_wrapper, xdata, ydata, bounds=bounds, p0=initial_guess)
    else:
        # Fix the transmission coefficient parameter and fit the model
        bounds = ((0.5, 0), (1, 1)) if bounds is None else bounds
        initial_guess = (0.9, 0.2)
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

def plot_r2_t15(axes):

    voltages = range(0, 5)
    filenames = [f"2024-2-27-ring/Radius2_t15_{v}v.csv" for v in voltages]

    deps = []
    phases = np.zeros(shape=(len(filenames), 2))


    for idx, filename in enumerate(filenames):

        df = pd.read_csv(filename)
        target_wavelength = 1550e-09
        wavelength = df["Wavelength"]
        ydata = df["Loss [dB]"] # this ydata should be positive.

        analysis = PAnalysis(wavelength, ydata, target_wavelength, cutoff=2, distance=100)
        fsr = analysis.fsr()

        xlow, xhigh = analysis.get_range_idx(range=1E-09)
        lambda_r = analysis.closest_resonance()

        phases[idx] = (lambda_r, fsr)

        wavelength = wavelength[xlow:xhigh]
        ydata = ydata[xlow:xhigh]
        axes[0].plot(wavelength*1E9, -ydata, label=f"{voltages[idx]}V")

        ydata = 10**(-ydata/10)
        
        if not idx:
            a, t, p = curve_fitting_for_ring(wavelength, ydata, 
                                        lambda_r=lambda_r, fsr=fsr)
            dep = t
            ind = a
            
        if idx:
            dep, p = curve_fitting_for_ring(wavelength, ydata, 
                                        lambda_r=lambda_r, fsr=fsr, fixed=ind)
        
        deps.append(dep)
        # print(f"Voltage: {voltages[idx]}, Amplitude: {dep}, Transmission: {ind}, Scaling by loss: {p}")

        ydata_fit = ring_resonator_model(df["Wavelength"], lambda_r, fsr, ind, dep)*p

        axes[1].plot(wavelength*1E9, ydata*1E+03, label=f"{voltages[idx]}V", color=f"C{idx}")
        axes[1].plot(wavelength*1E9, ydata_fit[xlow:xhigh]*1E+03, "--", color=f"C{idx}")
    

    axes[2].plot(voltages, deps, label="a", marker="o")
    # axes[2].plot(voltages, deps)
    axes[2].plot(voltages, [ind]*len(voltages), label="t", marker="o")
    axes[2].legend()
    axes[2].set_xlabel("Voltage [V]")
    axes[2].set_ylabel("Coefficient")
    axes[2].set_xlim(-0.1, 4.1)
    

    length = 2*np.pi*29*1E-06*0.83
    alpha = PAnalysis.get_loss_from_a(deps, length)
    alpha = alpha - alpha[0]
    phases[:,0] = phases[:,0] - phases[0,0]
    axes[3], ax2 = Publication.twin_x(axes[3])
    phase_shift = PAnalysis.get_phase_shift_from_rshift(phases[:, 0], phases[:, 1])
    modeff = PAnalysis.get_modeff_from_rshift(voltages, phases[:, 0], phases[:, 1], length)
    axes[3].plot(voltages, modeff*1E+03, marker="o")
    axes[3].set_xlabel("Voltage [V]")
    axes[3].set_ylabel("Modulation efficiency [Vmm]")
    axes[3].set_xticks(voltages[1:])
    ax2.plot(voltages, phase_shift/np.pi, marker="o", color="C1")
    ax2.set_ylabel(r"$\Delta \Phi$ [\pi]")
    ax2.set_xticks(voltages[1:])


    
    axes[0].legend()
    # axes[0].set_ylim(-63, -25)
    axes[0].set_xlabel("Wavelength [nm]")
    axes[0].set_ylabel("Loss [dB]")
    axes[0].set_xlim(1548.63, 1549.63)
    axes[0].set_xticks([1548.63, 1549.13, 1549.63])
    axes[1].legend()
    # axes[1].set_ylim(0, 2.5)
    axes[1].set_xlabel("Wavelength [nm]")
    axes[1].set_ylabel(r"Transmission [uW]")
    axes[1].set_xlim(1548.63, 1549.63)
    axes[1].set_xticks([1548.63, 1549.13, 1549.63])

def plot_r4_t15(axes):
    voltages = range(0, 5)
    filenames = [f"2024-2-27-ring/Radius4_t15_{v}v.csv" for v in voltages]

    deps = []
    phases = np.zeros(shape=(len(filenames), 2))

    for idx, filename in enumerate(filenames):

        df = pd.read_csv(filename)
        target_wavelength = 1550e-09
        wavelength = df["Wavelength"]
        ydata = df["Loss [dB]"] # this ydata should be positive.

        analysis = PAnalysis(wavelength, ydata, target_wavelength, cutoff=2, distance=100)
        fsr = analysis.fsr()

        xlow, xhigh = analysis.get_range_idx(range=1E-09)
        lambda_r = analysis.closest_resonance()

        phases[idx] = (lambda_r, fsr)

        wavelength = wavelength[xlow:xhigh]
        ydata = ydata[xlow:xhigh]
        axes[0].plot(wavelength*1E9, -ydata, label=f"{voltages[idx]}V")

        ydata = 10**(-ydata/10)
        
        if not idx:
            a, t, p = curve_fitting_for_ring(wavelength, ydata, 
                                        lambda_r=lambda_r, fsr=fsr)
            dep = a
            ind = t
            
        if idx:
            dep, p = curve_fitting_for_ring(wavelength, ydata, 
                                        lambda_r=lambda_r, fsr=fsr, fixed=ind)
        
        deps.append(dep)
        # print(f"Voltage: {voltages[idx]}, Amplitude: {dep}, Transmission: {ind}, Scaling by loss: {p}")

        ydata_fit = ring_resonator_model(df["Wavelength"], lambda_r, fsr, ind, dep)*p

        axes[1].plot(wavelength*1E9, ydata*1E+03, label=f"{voltages[idx]}V", color=f"C{idx}")
        axes[1].plot(wavelength*1E9, ydata_fit[xlow:xhigh]*1E+03, "--", color=f"C{idx}")
    

    axes[2].plot(voltages, deps, label="a", marker="o")
    # axes[2].plot(voltages, deps)
    axes[2].plot(voltages, [ind]*len(voltages), label="t", marker="o")
    axes[2].legend()
    axes[2].set_xlabel("Voltage [V]")
    axes[2].set_ylabel("Coefficient")
    axes[2].set_xlim(-0.1, 4.1)

    length = 2*np.pi*15*1E-06*0.83
    alpha = PAnalysis.get_loss_from_a(deps, length)
    alpha = alpha - alpha[0]
    print(alpha/1E+03)
    phases[:,0] = phases[:,0] - phases[0,0]
    axes[3], ax2 = Publication.twin_x(axes[3])
    phase_shift = PAnalysis.get_phase_shift_from_rshift(phases[:, 0], phases[:, 1])
    modeff = PAnalysis.get_modeff_from_rshift(voltages, phases[:, 0], phases[:, 1], length)
    axes[3].plot(voltages, modeff*1E+03, marker="o")
    axes[3].set_xlabel("Voltage [V]")
    axes[3].set_ylabel("Modulation efficiency [Vmm]")
    axes[3].set_xticks(voltages[1:])
    ax2.plot(voltages, phase_shift, marker="o", color="C1")
    ax2.set_ylabel(r"$\Delta \Phi$ [rad]")
    ax2.set_xticks(voltages[1:])



    axes[0].legend()
    # axes[0].set_ylim(-63, -25)
    axes[0].set_xlabel("Wavelength [nm]")
    axes[0].set_ylabel("Loss [dB]")
    axes[0].set_xlim(1552.35, 1553.35)
    axes[0].set_xticks([1552.35, 1552.85, 1553.35])
    axes[1].legend()
    # axes[1].set_ylim(0, 2.5)
    axes[1].set_xlabel("Wavelength [nm]", fontsize=16)
    axes[1].set_ylabel(r"Transmission [uW]")
    axes[1].set_xlim(1552.35, 1553.35)
    axes[1].set_xticks([1552.35, 1552.85, 1553.35])


def plot_ramzi(axes):
    filenames = [
        r"D:\2024-5-28\ramzi_g210\omr\g210_h1.59V_0v_max_max.omr",
        # r"D:\2024-5-28\ramzi_g210\omr\g210_h1.516V_1v_max_max.omr",
        # r"D:\2024-5-28\ramzi_g210\omr\g210_h1.974V_2v_max_max.omr",
        # r"D:\2024-5-28\ramzi_g210\omr\g210_h1.966V_3v_max_max.omr",
        # r"D:\2024-5-28\ramzi_g210\omr\g210_h0.2V_4v_max_max.omr"
    ]
    target_wavelength = 1552e-09
    for idx, filename in enumerate(filenames):

        df = Parser.iomr_parse(Path(filename))
        
        wavelength = df["Wavelength"]
        ydata = df["Loss [dB]"] # this ydata should be positive.
        axes[0].plot(wavelength, -ydata, label=f"{idx}V")

        analysis = PAnalysis(wavelength, ydata, target_wavelength, cutoff=4, distance=500)
        target_wavelength = analysis.closest_resonance()
        fsr = analysis.fsr()

        xlow, xhigh = analysis.get_range_idx(range=1E-09)
        lambda_r = analysis.closest_resonance()

        wavelength = wavelength[xlow:xhigh]
        ydata = ydata[xlow:xhigh]
        
        ydata = 10**(-ydata/10)
        axes[1].plot(wavelength*1E9, ydata, label=f"{idx}V")
        
        if not idx:     
            a, t, p = curve_fitting_for_ramzi(wavelength, ydata, lambda_r, fsr, 0)
            dep = a
            ind = t
        else:
            dep, p = curve_fitting_for_ramzi(wavelength, ydata, lambda_r, fsr, 0, fixed=ind)

        print(f"dep: {dep}, ind: {ind}, p: {p}")

        yfit = ramzi_output_intensity(wavelength, lambda_r, fsr, 0, a, t)*p
        axes[1].plot(wavelength*1E9, yfit, label=f"{idx}V", linestyle="--")


def main():
    Publication.set_basics()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()
    plot_r2_t15(axes[0:4])
    plot_r4_t15(axes[4:8])
    Publication.set_titles(axes, col=True)
    fig.tight_layout()
    fig.savefig("ring_resonator_fit.pdf")
    plt.show()

def main2():
    Publication.set_basics()
    fig, axes = plt.subplots(1, 2)
    axes = axes.flatten()
    plot_ramzi(axes)
    plt.show()

if __name__ == "__main__":
    main2()
