from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from postprocessing.utils.formatter import Publication
from postprocessing.analysis import PAnalysis
from postprocessing.parser import Parser
from postprocessing.fitting import (
    curve_fitting_for_ring,
    ring_resonator_model,
)
from postprocessing.conversion import a_to_db_alpha


def ring_fitting_1var():

    fig, axes = plt.subplots(1, 4)
    axes = axes.flatten()

    voltages = np.arange(0, 5, 1)
    deps = []
    phases = np.zeros(shape=(len(voltages), 2))

    target_wavelength = 1550
    directory = Path(r"C:\Users\tyc1g20\Downloads\2024-7-11\2024-7-11\D10")
    gap = 200

    for idx, volt in enumerate(voltages):

        filename = directory / f"{volt}v_g{gap}_r{volt}.omr"
        df = Parser.iomr_parse(Path(filename))
        wavelength = df["Wavelength"]
        ydata = df["Loss [dB]"] # this ydata should be positive.

        analysis = PAnalysis(wavelength, ydata, target_wavelength, cutoff=2, distance=100)
        target_wavelength = analysis.closest_resonance()
        fsr = analysis.fsr(num_peaks=2)
        
        xlow, xhigh = analysis.get_range_idx(xrange=1.5)
        lambda_r = analysis.closest_resonance()

        phases[idx] = (lambda_r, fsr)

        wavelength = wavelength[xlow:xhigh]
        ydata = ydata[xlow:xhigh]
        axes[0].plot(wavelength, -ydata, label=f"{voltages[idx]}V")
        ydata = 10**(-ydata/10)
        
        if not idx:
            a, t, p = curve_fitting_for_ring(wavelength, ydata, 
                                        lambda_r=lambda_r, fsr=fsr)
            dep = t
            ind = a
            
        if idx:
            dep, p = curve_fitting_for_ring(wavelength, ydata, 
                                        lambda_r=lambda_r, fsr=fsr, fixed_param=ind)
        
        deps.append(dep)
        print(f"Voltage: {voltages[idx]}, Amplitude: {dep}, Transmission: {ind}, Scaling by loss: {p}")

        ydata_fit = ring_resonator_model(df["Wavelength"], lambda_r, fsr, ind, dep)*p

        axes[1].plot(wavelength, ydata*1E+03, label=f"{voltages[idx]}V", color=f"C{idx}")
        axes[1].plot(wavelength, ydata_fit[xlow:xhigh]*1E+03, "--", color=f"C{idx}")
    
    length = 2*np.pi*15*1E-06*0.83
    axes[2].plot(voltages, deps, label="g=310nm,a", marker="o", color="C1")
    axes[2].plot(voltages, [ind]*len(voltages), label=r"g=310nm,$\gamma$", linestyle="--", marker="o",color="C1")
    axes[2].legend(loc=[0.05, 0.1])
    axes[2].set_xlabel("Voltage [V]")
    axes[2].set_ylabel("Coefficient")
    axes[2].set_xlim(-0.1, 4.1)
    axes[2].set_xticks(voltages)
    axes[2].set_yticks([0.93, 0.94, 0.95, 0.96])

    alpha = a_to_db_alpha(np.array(deps), length*1E+03)
    axes[3].plot(voltages, alpha - alpha[0], linestyle="--", label="g=310nm", marker="^")
    axes[3].set_ylabel(r"|$\Delta \alpha$| [dB/mm]")
    axes[3].set_xlim(-0.1, 4.1)
    axes[3].set_ylim(-0.1, 2.3)
    axes[3].set_xticks(voltages)
    axes[3].set_yticks(np.linspace(0, 2, 5))
    axes[3].legend(loc=[0.05, 0.35])
    # set yticks to have 5 labels
    # axes[3].set_yticks([])
    
    alpha = PAnalysis.get_loss_from_a(deps, length)
    alpha = alpha - alpha[0]
    phases[:,0] = phases[:,0] - phases[0,0]

    
    phase_shift = PAnalysis.get_phase_shift_from_rshift(phases[:, 0], phases[:, 1])
    modeff = PAnalysis.get_modeff_from_rshift(voltages, phases[:, 0], phases[:, 1], length)
    axes[4].plot(voltages, modeff*1E+03, marker="o", label="g=310nm")
    axes[4].set_xlabel("Voltage [V]")
    axes[4].set_ylabel("Modulation efficiency [Vmm]")
    axes[4].set_xticks(voltages[1:])

def main():
    Publication.set_basics()

    ring_fitting_1var()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
