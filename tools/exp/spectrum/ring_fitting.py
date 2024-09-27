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
    folder = Path(r"/path/to/folder")
    
    voltages = np.arange(1, 4.5, 0.5)
    target_wavelength = 1550E-09
    
    phases = []
    fig, ax = plt.subplots(1, 2)
    for idx, volt in enumerate(voltages):
        df = Parser.omr_parse(folder / f"ring{volt}V.omr")
        wavelength = df["Wavelength"]
        loss = df["Loss [dB]"]
        ax[0].plot(wavelength, -loss, label=f"{volt}V")
        
        analysis = PAnalysis(
            xdata=wavelength,
            ydata=loss,
            wavelength=target_wavelength
        )
        fsr = analysis.fsr(num_peaks=2)
        wres = analysis.true_res_wavelength
        
        # this get the indices within the xrange of the true resonance
        # for the fitting. Make sure xrange is large enough to capture
        # the whole or at least enough of the resonance
        xlow, xhigh = analysis.get_range_idx(xrange=1.5E-09)
        phases.append((wres, fsr))
        wavelength = wavelength[xlow:xhigh]
        loss = loss[xlow:xhigh]
        power = db2w(-loss)
        
        if idx == 0:
            a, t, p = curve_fitting_for_ring(wavelength, power, lambda_r=wres, fsr=fsr)
            
            # Will need to change these two values based on your knowledge of the data. Fixing one coefficient to a constant as coupling coefficient shouldn't change.
            dep = t
            ind = a
        else:
            dep, p = curve_fitting_for_ring(wavelength, power, lambda_r=wres, fsr=fsr, fixed_param=ind)

        print(f"{volt}V  -  Ind. (t): {ind}, Dep. (a): {dep}, P: {p}")
        loss_fit = ring_resonator_model(wavelength, wres, fsr, ind, dep) * p
        ax.plot(wavelength*1E+09, loss*1E+03, label=f"{volt}V", color=f"C{idx}")
        ax.plot(wavelength*1E+09, loss_fit*1E+03, linestyle="--", color=f"C{idx}")
      
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
    
    
    
        
        
        
        
        
    
    