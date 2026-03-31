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
    folder = Path(r"Z:\Christina\experiments\2026-01-27")
    
    couplings = [("31", "41")]
    number_of_tests = np.arange(0, 6)
    coupling_gaps = [320, 400, 280, 360, 240, 200]
    target_wavelength = 1556E-09
    
    ratios1, ratios2 = [], []
    fig, ax = plt.subplots(2, 2, figsize=(12, 5))
    ax = ax.flatten()
    for i in number_of_tests:
        for cs in couplings:
            df1 = Parser.omr_parse(folder / f"coupling{i}-{cs[0]}.omr", convert_to_csv=True)
            df2 = Parser.omr_parse(folder / f"coupling{i}-{cs[1]}.omr", convert_to_csv=True)

            analysis1 = PAnalysis(
                xdata=df1["Wavelength"],
                ydata=df1["Loss [dB]"],
                wavelength=target_wavelength
            )

            analysis2 = PAnalysis(
                xdata=df2["Wavelength"],
                ydata=df2["Loss [dB]"],
                wavelength=target_wavelength
            )

            w1 = df1["Wavelength"]
            l1 = df1["Loss [dB]"]
            w2 = df2["Wavelength"]
            l2 = df2["Loss [dB]"]
            ax[0].plot(w1, -l1, label=f"{cs[0]}V")
            ax[0].plot(w2, -l2, label=f"{cs[1]}V")
            ax[1].plot(w1, 10**(-l1/10), label=f"{cs[0]}")
            ax[1].plot(w2, 10**(-l2/10), label=f"{cs[1]}")
            power = 0.35817890134
            # ratio1 = 10**(-l1/10) / (10**(-l1/10)+10**(-l2/10))
            # ratio2 = 10**(-l2/10) / (10**(-l1/10)+10**(-l2/10))
            ratio1 = 10**(-l2/10) / power
            ratio2 = 10**(-l2/10) / power
            ax[2].plot(w1, ratio1, label=f"{cs[0]}")
            ax[2].plot(w1, ratio2, label=f"{cs[1]}")

            power1 = 10**(-analysis1.ydata[analysis1.target_wavelength_idx]/10)
            power2 = 10**(-analysis2.ydata[analysis1.target_wavelength_idx]/10)
            
            ratios1.append(power1 / (power1 + power2))
            ratios2.append(power2 / (power1 + power2))
            # analysis = PAnalysis(
            #     xdata=wavelength,
            #     ydata=loss,
            #     wavelength=target_wavelength
            # )
            # fsr = analysis.fsr(num_peaks=2)
            # wres = analysis.true_res_wavelength
    

    print(ratios1, ratios2)
    joint_list = list(zip(*sorted(zip(coupling_gaps, ratios1))))
    joint_list2 = list(zip(*sorted(zip(coupling_gaps, ratios2))))
    
    ax[3].plot(joint_list[0], joint_list[1])
    ax[3].plot(joint_list2[0], joint_list2[1])
      
    ax[0].set_xlabel("Wavelength [nm]")
    ax[0].set_ylabel("Loss [dB]")
    ax[0].legend()  
    ax[1].set_xlabel("Wavelength [nm]")
    ax[1].set_ylabel("Transmission power [mW]")
    ax[1].legend()
    ax[2].set_xlabel("Wavelength [nm]")
    ax[2].set_ylabel("Ratio")
    ax[2].legend()
    
    # fig.savefig("ring_fit.png", dpi=400)
    
    plt.show()
    
if __name__ == "__main__":
    main()
