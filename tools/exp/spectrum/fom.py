"""
fom.py

This script calculates the modulation efficiency based on phase shift of the spectrum.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.utils.formatter import Publication
from postprocessing.parser import Parser
from postprocessing.analysis import PAnalysis

def main():
    Publication.set_basics()

    folder = Path(r"/path/to/folder")
    
    voltages = np.arange(1, 4.5, 0.5)
    length = 2*np.pi*15*1E-06*0.83
    
    fig, ax = plt.subplots(1, 2)
    phi = []
    for volt in voltages:
        df = Parser.omr_parse(folder / f"ring{volt}V.omr")
        analysis = PAnalysis(xdata=df["Wavelength"], ydata=df["Loss [dB]"], wavelength=1550E-09)
        phi.append(analysis.true_res_wavelength, analysis.fsr(num_peaks=2))
    
    phase_shift = PAnalysis.get_phase_shift_from_rshift(phi[:, 0], phi[:, 1])
    mod_eff = PAnalysis.get_modeff_from_rshift(voltages, phi[:, 0], phi[:, 1], length)
    
    ax[0].plot(voltages, phase_shift/np.pi)
    ax[0].set_xlabel("Biased Voltage [V]")
    ax[0].set_ylabel(r"|$\Delta \Phi|$ $[\pi]$")
    
    ax[1].plot(voltages, mod_eff)
    ax[1].set_xlabel("Voltage [V]")
    ax[1].set_ylabel("Modulation Efficiency [V.mm]")
    
    # fig.savefig("mod_eff.png", dpi=400)
    plt.show()
    
if __name__ == "__main__":
    main()
        