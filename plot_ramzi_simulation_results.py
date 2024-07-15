from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from postprocessing.parser import Parser
from postprocessing.utils.formatter import Publication

def plot_ramzi_coefficients():
    Publication.set_basics()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    filename = Path(r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\ring-assisted moscap\ring_mzi\investigate_coeffs\ramzi_16um_0.8_sweep_different_ring_coupling_coefficient.mat")
    data = Parser.matlab_parse(filename)
    print(data.keys())
    voltages = np.arange(0, 3, 1)
    for v in voltages:
        ax[0].plot(data[f"lum.x{v}"]*1E+09, data[f"lum.y{v}"])

    filename = Path(r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\ring-assisted moscap\ring_mzi\investigate_coeffs\ramzi_16um_0.8_sweep_different_splitting_ratio.mat")
    data = Parser.matlab_parse(filename)
    voltages = np.arange(0, 4, 1)
    print(data.keys())
    for v in voltages:
        ax[1].plot(data[f"lum.x{v}"], data[f"lum.y{v}"], label=f"{v}V")
    
    ax[0].set_xlim(1552.5, 1554)
    ax[0].set_ylim(-11, 0)
    ax[0].set_xticks(np.linspace(1553, 1554, 3))
    ax[0].set_xlabel("Wavelength [nm]")
    ax[0].set_ylabel("Transmission [dB]")
    ax[0].legend(["under", "critical", "over"], loc="lower right")

    ax[1].set_xlim(1552.5, 1554)
    ax[1].set_ylim(-50, 0)
    ax[1].set_xticks(np.linspace(1553, 1554, 3))
    ax[1].set_xlabel("Wavelength [nm]")
    ax[1].set_ylabel("Transmission [dB]")
    ax[1].legend(["30%", "50%", "88%", "95%"], loc="lower right")

    ax[0].set_title("(a)")
    ax[1].set_title("(b)")
    fig.tight_layout()
    fig.savefig("ramzi_splitting_simulation_results.pdf")
    plt.show()

def plot_ramzi_and_ring_spectrum():
    Publication.set_basics()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    filename = Path(r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\ring-assisted moscap\ring_mzi\investigate_coeffs\single_ring_16um_0.8.mat")
    data = Parser.matlab_parse(filename)
    voltages = np.arange(0, 7, 1)
    for v in voltages:
        ax[0].plot(data[f"lum.x{v}"]*1E+09, data[f"lum.y{v}"], label=f"{v}V")
    
    ax[0].set_xlabel("Wavelength [nm]")
    ax[0].set_ylabel("Transmission [dB]")
    ax[0].set_xticks(np.linspace(1552.5, 1554, 4))
    ax[0].set_xlim(1552.2, 1554)
    ax[0].set_ylim(-50, 0)

    filename = Path(r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\ring-assisted moscap\ring_mzi\investigate_coeffs\ramzi_16um_0.8_coeff_0.88.mat")
    data = Parser.matlab_parse(filename)
    for v in voltages:
        ax[2].plot(data[f"lum.x{v}"]*1E+09, data[f"lum.y{v}"], label=f"{v}V")

    ax[2].set_xlim(1552.2, 1554)
    ax[2].set_ylim(-50, 0)
    ax[2].set_xticks(np.linspace(1552.5, 1554, 4))
    ax[2].set_xlabel("Wavelength [nm]")
    ax[2].set_ylabel("Transmission [dB]")

    filename = Path(r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\ring-assisted moscap\ring_mzi\investigate_coeffs\ramzi_16um_0.8_50_split.mat")
    data = Parser.matlab_parse(filename)
    for v in voltages:
        ax[1].plot(data[f"lum.x{v}"]*1E+09, data[f"lum.y{v}"], label=f"{v}V")

    ax[1].set_xlim(1552.2, 1554)
    ax[1].set_ylim(-50, 0)
    ax[1].set_xticks(np.linspace(1552.5, 1554, 4))
    ax[1].set_xlabel("Wavelength [nm]")
    ax[1].set_ylabel("Transmission [dB]")

    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower right")
    ax[2].legend(loc="lower right")
    ax[0].set_title("(a)")
    ax[1].set_title("(b)")
    ax[2].set_title("(c)")
    fig.tight_layout()
    fig.savefig("ramzi_simulation_results.pdf")
    plt.show()


if __name__ == "__main__":
    plot_ramzi_coefficients()