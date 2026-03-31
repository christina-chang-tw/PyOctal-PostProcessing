from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def main():
    voltages = [0, 1, 2, 3, 4, 5, 6]
    folder = Path(r"C:\Users\tyc1g20\Downloads")
    file = folder / f"spectrum.csv"
    df = pd.read_csv(file)
    wavelength = df["Wavelength (nm)"]


    for volt in voltages:        
        plt.plot(wavelength, 10 ** (df[f"{volt}V"]/10), label=f"{volt}V")

    plt.ylim(0, 0.5)
    plt.legend()
    plt.xlabel("Wavelength (nm)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.8)
    # plt.ylabel("Loss (dB)")
    plt.ylabel("Power (W)")
    plt.show()

if __name__ == "__main__":
    main()
