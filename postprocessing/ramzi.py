from os.path import join
from os import listdir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from postprocessing.analysis import PAnalysis
from postprocessing.utils.util import averaging

def get_voltage_against_power(data: dict, idx: int):
    """
    Calculate power at specified wavelength.

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
    voltages = np.arange(0, 1.5, 0.01)
    ring_vs = np.arange(0, 3.5, 0.5)
    target_wavelength = 1550e-09

    files = listdir(folder)
    print(files[:5])

    data = {}

    fig, ax = plt.subplots(1, 1)

    for ring_v in ring_vs:
        for volt in voltages:
            file = join(folder, f"ring{ring_v:.1f}V_heat{volt:.2f}V.csv")
            df = pd.read_csv(file)

            ydata = df["Loss [dB]"]

            if volt == voltages[0]:
                xdata = df["Wavelength"]
                analysis = PAnalysis(xdata, ydata, target_wavelength, cutoff=10, distance=100)
                index = analysis.target_resonance_idx

            data[volt] = - ydata

        vs, powers = get_voltage_against_power(data, index)
        ax.plot(vs, averaging(powers, 5), label=f"Ring {ring_v:.1f}V", linestyle="--")
        # ax.plot(vs, powers, label=f"Ring {ring_v:.1f}V")
    
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()