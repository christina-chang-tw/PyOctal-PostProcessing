import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

from postprocessing.utils.formatter import Publication


def convert_freq_to_wavelength(freq):
    return 299792458 / freq

def fit_coupling_coefficient(xdata, ydata):
    func = interp1d(xdata, ydata, kind="cubic", fill_value="extrapolate")
    new_xdata = np.linspace(xdata.min(), xdata.max(), 10000)
    return new_xdata, func

def coupling_1var_with_cvw():
    directory = Path(r"C:\Users\tyc1g20\Downloads\txt")
    cl = 8
    radii = np.arange(5, 31, 1)

    couplings = []
    gap = 250
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    for r in radii:
        filename = directory / f"14_07_2024_cs_cl{cl}um_{r}um_gap{gap}nm.txt"
        data = np.loadtxt(filename, delimiter=" ")
        wavelength = convert_freq_to_wavelength(data[:, 0])
        new_x, func = fit_coupling_coefficient(wavelength, data[:, 1])
        
        couplings = np.append(couplings, func(1.55E-06))

        ax[0].plot(new_x*1E+09, func(new_x), label=f"{gap}nm")

    ax[1].plot(radii, np.sqrt(couplings), color="C0", marker="o")
    ax[1], ax2 = Publication.twin_x(ax[1])
    ax2.plot(radii, np.sqrt(1-couplings), color="C1", marker="o")
    
    ax[0].set_xlim(1545, 1555)
    ax[0].legend()
    ax[0].set_xlabel("Wavelength [nm]")
    ax[0].set_ylabel(r"Cross coupling coefficient ($\kappa^2$)")
    ax[1].set_xlabel("Radius [um]")
    ax[1].set_ylabel("Cross coupling coefficient @ 1550nm")
    ax2.set_ylabel("Through coupling coefficient @ 1550nm")

def coupling_1var():
    directory = Path(r"C:\Users\tyc1g20\Downloads\txt")
    cl = 8
    radii = np.arange(5, 31, 1)

    couplings = []
    gap = 250
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    for r in radii:
        filename = directory / f"14_07_2024_cs_cl{cl}um_{r}um_gap{gap}nm.txt"
        data = np.loadtxt(filename, delimiter=" ")
        wavelength = convert_freq_to_wavelength(data[:, 0])
        new_x, func = fit_coupling_coefficient(wavelength, data[:, 1])
        
        couplings = np.append(couplings, func(1.55E-06))

    ax[0].plot(radii, np.sqrt(couplings), marker="o")
    ax[1].plot(radii, np.sqrt(1-couplings), marker="o")
    
    ax[0].legend()
    ax[0].set_xlabel("Radius [um]")
    ax[0].set_ylabel("Cross coupling coefficient @ 1550nm")
    ax[1].legend()
    ax[1].set_xlabel("Radius [um]")
    ax[1].set_ylabel("Self coupling coefficient @ 1550nm")

def coupling_2vars():
    directory = Path(r"C:\Users\tyc1g20\Downloads\txt")
    gaps = np.arange(200, 300, 10) # first param
    gap = 250
    radii = np.arange(10, 25, 5)
    cls = np.arange(5, 16, 1)
    cl = 8
    
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    for radius in radii:
        couplings = []
        for gap in gaps:
            filename = directory / f"12_07_2024_cs_cl{cl}um_r{radius}um_gap{gap}nm.txt"
            data = np.loadtxt(filename, delimiter=" ")
            wavelength = convert_freq_to_wavelength(data[:, 0])
            _, func = fit_coupling_coefficient(wavelength, data[:, 1])
            
            couplings = np.append(couplings, func(1.55E-06))

        ax[0].plot(gaps, np.sqrt(couplings), marker="o", label=f"r={radius}um")
        ax[1].plot(gaps, np.sqrt(1-couplings), marker="o", label=f"r={radius}um")
    
    ax[0].legend()
    ax[0].set_xlabel("Gap [nm]")
    ax[0].set_ylabel("Cross coupling coefficient @ 1550nm")
    ax[1].legend()
    ax[1].set_xlabel("Gap [nm]")
    ax[1].set_ylabel("Self coupling coefficient @ 1550nm")

def main():
    Publication.set_basics()
    coupling_1var()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
