from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from postprocessing.utils.formatter import Publication

# Load the data

def plot_coupling_gap():
    fig, ax = plt.subplots(1, 1)
    radius = [8, 10, 15, 20]

    for i, radii in enumerate(radius):
        files = list(Path(r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\ring-assisted moscap\coupling_coeffs\{}um".format(radii)).rglob("*.txt"))
        values_at_1550 = np.zeros(shape=(len(files),2))
        for idx, file in enumerate(files):
            match = re.search(r'g(\d+\.?\d*e[+-]\d+)', file.stem)
            value = float(match.group(1))


            data = np.loadtxt(file)
            wavelength = 299792458/data[:, 0]*1e9
            # cubic spline interpolation
            interp_func = interp1d(wavelength, data[:, 1], kind='cubic')
            y_1550 = interp_func(1550)
            new_wavelength = np.linspace(wavelength[0], wavelength[-1], 10000)
            new_data = interp_func(new_wavelength)
            values_at_1550[idx] = (value, y_1550)

        values_at_1550 = values_at_1550[values_at_1550[:, 0].argsort()]

        ax.plot(values_at_1550[:, 0]*1E+06, 1-values_at_1550[:, 1], label=f"R={radii}um")
        ax.set_xlabel("Gap [um]")
        ax.set_ylabel(r"$|S_{11}|^2$")
        ax.legend()
        ax.set_xlim(0.2, 0.3)
        ax.set_ylim(0.55, 1)

    fig.savefig("coupling_gap.png", dpi=400)
    fig.tight_layout()
    plt.show()

def plot_coupling_gap2():
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    radius = [8, 10, 15, 20]

    for i, radii in enumerate(radius):
        files = list(Path(f"{radii}um").rglob("*.txt"))
        values_at_1550 = np.zeros(shape=(len(files),2))
        for idx, file in enumerate(files):
            match = re.search(r'g(\d+\.?\d*e[+-]\d+)', file.stem)
            value = float(match.group(1))


            data = np.loadtxt(file)
            wavelength = 299792458/data[:, 0]*1e9
            # cubic spline interpolation
            interp_func = interp1d(wavelength, data[:, 1], kind='cubic')
            y_1550 = interp_func(1550)
            new_wavelength = np.linspace(wavelength[0], wavelength[-1], 10000)
            new_data = interp_func(new_wavelength)
            values_at_1550[idx] = (value, y_1550)

        values_at_1550 = values_at_1550[values_at_1550[:, 0].argsort()]

        ax[i].plot(values_at_1550[:, 0]*1E+06, values_at_1550[:, 1], label=f"R={radii}um")
        ax[i].set_xlabel("Gap [um]")
        ax[i].set_ylabel("Coupling Coefficient |S21|^2")
        ax[i].legend()

        alpha_0 = 2
        alpha_1 = 3
        # convert db to non-db
        alpha_0 = 10**(alpha_0/10)
        alpha_1 = 10**(alpha_1/10)
        length = 2*np.pi*radii*0.8*1E-03
        # draw a straight line on the plot with y = alpha
        ax[i].axhline(y=1-np.exp(-alpha_0*length), color='r', linestyle='--')
        ax[i].axhline(y=1-np.exp(-alpha_1*length), color='b', linestyle='--')

    fig.savefig("coupling_gap.png", dpi=400)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    Publication.set_basics()
    plot_coupling_gap()