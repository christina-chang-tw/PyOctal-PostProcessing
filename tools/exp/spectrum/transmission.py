"""
transmission.py

This script is used to plot the transmission spectrum of a ring resonator for different biasing conditions.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from postprocessing.parser import Parser
from postprocessing.utils.formatter import Publication

def main():
    Publication.set_basics()
    folder = Path(r"/path/to/folder")
    voltages = [0, 1, 2, 3, 4]
    
    fig, ax = plt.subplots(1, 1)
    
    for volt in voltages:
        filename = folder / f"ring{volt}V.omr"
        # convert to a csv file speeds up the access
        df = Parser.omr_parse(filename)
        ax.plot(df["Wavelength"] * 1E+09, -df["Loss [dB]"], label=f"{volt}V")
        
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Loss [dB]")
    ax.legend()
    
    # fig.savefig("spectrum.png", dpi=400) # save the fig
    plt.show()
    
if __name__ == "__main__":
    main()    
