"""
ring_fitting.py

This script is used to fit a ring resonator model to a set of ring resonator data. The data is assumed to be in the form of a .omr file. It will fit the ring resonator model to the data and output the coupling coefficients and the power.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.utils.formatter import Publication
from postprocessing.analysis import PAnalysis
from postprocessing.parser import Parser
from postprocessing.fitting import (
    curve_fitting_for_ring,
    ring_resonator_model,
)
from postprocessing.utils.conversion import db2w

def main():
    Publication.set_basics()
    folder = Path(r"/Z:\Christina\experiments\2026-01-27")
    
    couplings = ["31", "32", "41", "42"]
    target_wavelength = 1550E-09
    
    phases = []
    fig, ax = plt.subplots(1, 2)
    for idx, c in enumerate(couplings):
        df = Parser.omr_parse(folder / f"coupling0-{c}.omr")
        wavelength = df["Wavelength"]
        loss = df["Loss [dB]"]
        ax[0].plot(wavelength, -loss, label=f"{c}V")
        ax[1].plot(wavelength, 10**(-loss/10), label=f"{c}V")
        
        # analysis = PAnalysis(
        #     xdata=wavelength,
        #     ydata=loss,
        #     wavelength=target_wavelength
        # )
        # fsr = analysis.fsr(num_peaks=2)
        # wres = analysis.true_res_wavelength
        
        
      
    ax[0].set_xlabel("Wavelength [nm]")
    ax[0].set_ylabel("Loss [dB]")
    ax[0].legend()  
    ax[1].set_xlabel("Wavelength [nm]")
    ax[1].set_ylabel("Transmission power [mW]")
    ax[1].legend()
    
    # fig.savefig("ring_fit.png", dpi=400)
    
    plt.show()
    
if __name__ == "__main__":
    main()
