"""
spectrum.py

Performing bandwidth analysis on spectrum extracted from LCA.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.parser import Parser
from postprocessing.utils.formatter import Publication
from postprocessing.utils.op import window_averaging, normalise

def main():
    Publication.set_basics() # my personal formatting tool
    folder = Path(r"path/to/omr/folder")
    
    voltages = np.arange(1, 4.5, 0.5)
    fig, ax = plt.subplots(1, 1)

    for volt in voltages:
        filename = folder / f"omr_{volt:.1f}V.s2p"
        df = Parser.snp_parse(filename)
        
        freq = df["freq"]
        s21 = df["s21"]
        s21 = normalise(freq, s21, 1e09)
        # Is it noisy? might need some averaging
        # s21 = window_averaging(s21, 100)
        
        # Might need to use this if the 3db point is incorrect. They set the upper and lower boundary of the frequency range. Uncomment if needed.
        # index = np.where((df["freq"] > 1e09) & (df["freq"] < 55e09))[0]
        # idx_3db = np.abs(s21[index[0]:index[-1]] + 3).argmin() + index[0]

        idx_3db = np.abs(s21 + 3).argmin()
        print(f"Voltage: {volt}V, 3dB [GHz]: {freq[idx_3db] * 1E-09}")
        
        ax.plot(freq*1E-09, s21, label=f"{volt}V")
    
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("s21 [dB]")
    ax.legend()
    fig.tight_layout()
    # fig.savefig("spectrum.png", dpi=400) # save the fig
     
    plt.show()
    
if __name__ == "__main__":
    main()