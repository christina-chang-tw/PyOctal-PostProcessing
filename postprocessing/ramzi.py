from os.path import join

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
    folder = r"2024-4-19-ring-assisted-mzi/2024-4-19-ring-assisted-mzi-csv"
    voltages = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    target_wavelength = 1550e-09

    data = {}

    for voltage in voltages:
        file = join(folder, f"ring{voltage}V_heat0.00V.csv")
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