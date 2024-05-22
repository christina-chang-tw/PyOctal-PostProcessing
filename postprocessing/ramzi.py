from os.path import join
from os import listdir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from postprocessing.analysis import PAnalysis

def get_voltage_against_power(data: dict, idx: int):
    """
    Calculate the absorption at the ring level one.

    Parameters:
        data: A list of ydata values
        idx: The index to get the power from.
    """
    powers = []

    for val in data.values():
        powers.append(val[idx])

    return data.keys(), powers


def main():

    # Load the data
    folder = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\ring-assisted moscap\ring_mzi\voltage_sweep_result_22_04_24"
    voltages = [0, 1, 2, 3, 4, 5, 6]
    target_wavelength = 1550e-09

    files = listdir(folder)
    print(files[:5])

    data = {}

    for voltage in voltages:
        file = join(folder, f"ring{voltage}_ph0.0V.csv")
        print(file)
        df = pd.read_csv(file)

        ydata = df["Loss [dB]"]

        if voltage == voltages[0]:
            xdata = df["Wavelength"]
            analysis = PAnalysis(xdata, ydata, target_wavelength, cutoff=10, distance=100)
            index = analysis.target_resonance_idx

        data[voltage] = - ydata

    voltages, powers = get_voltage_against_power(data, index)
    plt.plot(voltages, powers)
    plt.show()

if __name__ == "__main__":
    main()