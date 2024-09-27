"""
coupling.py

This script is used to plot the coupling coefficients for a set of ring resonators. The data is assumed to be in the form of a .txt file. It will plot the cross and self coupling coefficients for the set of ring resonators.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from postprocessing.utils.formatter import Publication
 
def main():
    Publication.set_basics()
    folder = Path(r"path/to/folder")
    vars = np.arange(5, 31, 1)
    
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    for var in vars:
        filename = folder / f"{var}.txt"
        data = np.loadtxt(filename, delimiter=" ")
        wavelength = 299792458 / data[:, 0]
        func = interp1d(wavelength, data[:, 1], kind="cubic", fill_value="extrapolate")
        
        couplings = np.append(couplings, func(1.55E-06))

    ax[0].plot(vars, np.sqrt(couplings), marker="o")
    ax[1].plot(vars, np.sqrt(1-couplings), marker="o")
    
    ax[0].legend()
    ax[0].set_xlabel("Radius [um]")
    ax[0].set_ylabel("Cross coupling coefficient @ 1550nm")
    ax[1].legend()
    ax[1].set_xlabel("Radius [um]")
    ax[1].set_ylabel("Self coupling coefficient @ 1550nm")
    
    # fig.savefig("coupling_coeff", dpi=400)
    plt.show()
    
if __name__ == "__main__":
    main()