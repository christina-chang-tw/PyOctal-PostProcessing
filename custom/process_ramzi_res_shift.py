import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from postprocessing.utils.formatter import Publication
from postprocessing.analysis import PAnalysis

def process_ramzi_res_shift(ax, filepath: Path):
    df = pd.read_csv(filepath)
    wavelength = df.pop("Wavelength")

    for idx, (key, val) in enumerate(df.items()):
        if idx % 10 == 0:
            ax.plot(wavelength, -val, label=key)
    

def main():
    Publication.set_basics()
    filepath = Path(r"D:\2024-6-04\ramzi_g200\mapping\ring1.0.csv")
    fig, ax = plt.subplots(1, 1)
    process_ramzi_res_shift(ax, filepath)
    ax.set_xlim(1545, 1555)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()