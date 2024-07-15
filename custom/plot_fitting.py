from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from postprocessing.utils.formatter import Publication
from postprocessing.analysis import PAnalysis
from postprocessing.parser import Parser
from postprocessing.fitting import (
    curve_fitting_for_ramzi, 
    ramzi_output_intensity,
    curve_fitting_for_ring,
    ring_resonator_model,
)
from postprocessing.conversion import a_to_db_alpha


def plot_r4_t9(axes, twinx):

    voltages = np.arange(0, 5, 1)
    deps = []
    phases = np.zeros(shape=(len(voltages), 2))

    target_wavelength = 1550
    for idx, volt in enumerate(voltages):

        filename = r"C:\Users\cchan\Downloads\2024-6-10\ring_r15um_t9_2\csv\g230_ring{:.1f}V.csv".format(volt)
        df = pd.read_csv(Path(filename))
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
        # print(f"Voltage: {voltages[idx]}, Amplitude: {dep}, Transmission: {ind}, Scaling by loss: {p}")

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
    # axes[4].plot(voltages, modeff*1E+03, marker="o", label="g=310nm")
    # axes[4].set_xlabel("Voltage [V]")
    # axes[4].set_ylabel("Modulation efficiency [Vmm]")
    # axes[4].set_xticks(voltages[1:])
    twinx.plot(voltages, phase_shift/np.pi, linestyle="--", marker="^", color="C1", label="g=310nm")
    twinx.set_ylim(-0.04, 0.005)
    twinx.set_ylabel(r"|$\Delta \Phi|$ $[\pi]$")
    twinx.set_yticks(np.linspace(0, -0.04, 5))

    df1 = pd.read_csv("ring_r15um_t9.csv")
    wavelength = df1.pop("Wavelength")
    lefts, rights = [], []
    for _, data in df1.items():
        left, right = PAnalysis.operating_region(wavelength, data, 0.2)
        lefts.append(left)
        rights.append(right)
    axes[4].plot(df1.columns, lefts, label="g=310nm", marker="o")
    axes[4].plot(df1.columns, rights, label="g=310nm", marker="^")

    axes[5].plot(wavelength, df1["2.0"].values, label="g=310nm")
    
    axes[5].set_ylabel("OMA")
    axes[5].set_xlabel("Wavelength [nm]")
    axes[5].set_xlim(-0.5, 0.5)
    
    axes[0].legend()
    # axes[0].set_ylim(-63, -25)
    axes[0].set_xlabel("Wavelength [nm]")
    axes[0].set_ylabel("Loss [dB]")
    axes[0].set_xlim(1552.7, 1553.5)
    axes[0].set_xticks([1552.7, 1553.1, 1553.5])
    axes[1].legend()
    # axes[1].set_ylim(0, 2.5)
    axes[1].set_xlabel("Wavelength [nm]")
    axes[1].set_ylabel(r"Transmission [$\mu$W]")
    axes[1].set_xlim(1552.7, 1553.5)
    axes[1].set_xticks([1552.7, 1553.1, 1553.5])

def plot_r4_t16(axes, twinx):

    voltages = np.arange(0, 5, 1)

    deps = []
    phases = np.zeros(shape=(len(voltages), 2))
    oma_ydata = []

    target_wavelength = 1550
    for idx, volt in enumerate(voltages):

        filename = r"C:\Users\cchan\Downloads\2024-6-10\ring_r15um_t16\csv\g230_ring{:.1f}V.csv".format(volt)
        df = pd.read_csv(Path(filename))
        
        wavelength = df["Wavelength"]
        ydata = df["Loss [dB]"] # this ydata should be positive.

        analysis = PAnalysis(wavelength, ydata, target_wavelength, cutoff=10, distance=100)
        target_wavelength = analysis.closest_resonance()
        fsr = analysis.fsr(num_peaks=3)
        
        xlow, xhigh = analysis.get_range_idx(xrange=1.5)
        lambda_r = analysis.closest_resonance()

        phases[idx] = (lambda_r, fsr)

        wavelength = wavelength[xlow:xhigh]
        ydata = ydata[xlow:xhigh]
        if volt in [1, 3]:
            oma_ydata.append(ydata)
        axes[0].plot(wavelength, -ydata, label=f"{voltages[idx]}V")

        ydata = 10**(-ydata/10)        
        
        if not idx:
            a, t, p = curve_fitting_for_ring(wavelength, ydata, 
                                        lambda_r=lambda_r, fsr=fsr)
            dep = t
            ind = a
            
        if idx:
            dep, p = curve_fitting_for_ring(wavelength, ydata, 
                                        lambda_r=lambda_r, fsr=fsr, fixed_param=ind, bounds=([0.88, 0], [1, 1]))
        
        deps.append(dep)
        # print(f"Voltage: {voltages[idx]}, Amplitude: {dep}, Transmission: {ind}, Scaling by loss: {p}")

        ydata_fit = ring_resonator_model(df["Wavelength"], lambda_r, fsr, ind, dep)*p

        axes[1].plot(wavelength, ydata*1E+03, label=f"{voltages[idx]}V", color=f"C{idx}")
        axes[1].plot(wavelength, ydata_fit[xlow:xhigh]*1E+03, "--", color=f"C{idx}")
    
    length = 2*np.pi*15*1E-06*0.83

    axes[2].plot(voltages, deps, label="g=240nm,a", marker="o", color="C0")
    axes[2].plot(voltages, [ind]*len(voltages), label=r"g=240nm,$\gamma$", linestyle="--", marker="o", color="C0")
    axes[2].legend()
    axes[2].set_xlabel("Voltage [V]")
    axes[2].set_ylabel("Coefficient")
    
    
    alpha = a_to_db_alpha(np.array(deps), length*1E+03)
    axes[3].plot(voltages, alpha-alpha[0], linestyle="--", marker="o", label="g=240nm", color="C0")
    axes[3].set_ylabel(r"|$\Delta \alpha$| [dB/mm]")

    
    alpha = PAnalysis.get_loss_from_a(deps, length)
    alpha = alpha - alpha[0]
    phases[:,0] = phases[:,0] - phases[0,0]
    
    phase_shift = PAnalysis.get_phase_shift_from_rshift(phases[:, 0], phases[:, 1])
    modeff = PAnalysis.get_modeff_from_rshift(voltages, phases[:, 0], phases[:, 1], length)
    # axes[4].plot(voltages, modeff*1E+03, marker="o", label="g=240nm")
    # axes[4].set_xlabel("Voltage [V]")
    # axes[4].set_ylabel("Modulation efficiency [Vmm]")
    # axes[4].set_xticks(voltages[1:])
    twinx.plot(voltages, phase_shift/np.pi, marker="o", color="C1")
    twinx.set_ylabel(r"|$\Delta \Phi|$ [\pi]")
    twinx.set_xticks(voltages[1:])

    df1 = pd.read_csv("ring_r15um_t16.csv")
    wavelength = df1.pop("Wavelength")
    lefts, rights = [], []
    for _, data in df1.items():
        left, right = PAnalysis.operating_region(wavelength, data, 0.2)
        lefts.append(left)
        rights.append(right)
    axes[4].plot(df1.columns, lefts, label="Left", marker="o")
    axes[4].plot(df1.columns, rights, label="Right", marker="o")



        
    axes[5].plot(wavelength, df1["2.0"], label="g=240nm")
    axes[5].set_xlim(-1, 1)
    
    axes[0].legend()
    # axes[0].set_ylim(-63, -25)
    axes[0].set_xlabel("Wavelength [nm]")
    axes[0].set_ylabel("Loss [dB]")
    axes[0].set_xlim(1549.2, 1550.0)
    axes[0].set_xticks([1549.2, 1549.6, 1550.0])
    axes[1].legend()
    # axes[1].set_ylim(0, 2.5)
    axes[1].set_xlabel("Wavelength [nm]")
    axes[1].set_ylabel(r"Transmission [$\mu$W]")
    axes[1].set_xlim(1549.2, 1550.0)
    axes[1].set_xticks([1549.2, 1549.6, 1550.0])


def plot_ramzi(axes):
    voltages = np.arange(0, 5, 1)

    target_wavelength = 1550e-09
    for idx, volt in enumerate(voltages):
        filename = r"C:\Users\cchan\Downloads\2024-6-10\ramzi_g200\{}v_g200_max.omr".format(volt)
        df = Parser.iomr_parse(Path(filename))
        
        wavelength = df["Wavelength"]
        ydata = df["Loss [dB]"] # this ydata should be positive.
        axes[0].plot(wavelength, -ydata, label=f"{idx}V")

        analysis = PAnalysis(wavelength, ydata, target_wavelength, cutoff=4, distance=500)
        target_wavelength = analysis.closest_resonance()
        fsr = analysis.fsr()

        xlow, xhigh = analysis.get_range_idx(xrange=0.7E-09)
        lambda_r = analysis.closest_resonance()

        wavelength = wavelength[xlow:xhigh]
        ydata = ydata[xlow:xhigh]
        
        ydata = 10**(-ydata/10)
        axes[1].plot(wavelength*1E9, ydata, label=f"{idx}V", color=f"C{idx}")
        
        if not idx:     
            a, t, p = curve_fitting_for_ramzi(wavelength, ydata, lambda_r, fsr, 0)
            dep = a
            ind = t
        else:
            dep, p = curve_fitting_for_ramzi(wavelength, ydata, lambda_r, fsr, 0, fixed_param=ind)

        print(f"dep: {dep}, ind: {ind}, p: {p}")

        yfit = ramzi_output_intensity(wavelength, lambda_r, fsr, 0, a, t)*p
        axes[1].plot(wavelength*1E9, yfit, label=f"{idx}V", linestyle="--", color=f"C{idx}")

def main2():
    Publication.set_basics()

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    axes = axes.flatten()
    axes[3], twinx = Publication.twin_x(axes[3])
    for ax in axes:
        ax.ticklabel_format(useOffset=False)

    plot_r4_t16([axes[0], axes[1], axes[2], axes[3], axes[7], axes[6]], twinx)
    plot_r4_t9([axes[4], axes[5], axes[2], axes[3], axes[7], axes[6]], twinx)
    
    axes[7].legend()
    axes[6].legend(loc=(0.5, 0.8))
    Publication.set_titles(axes, col=True)
    fig.tight_layout()
    fig.savefig("ring_comparison.pdf")

    plt.show()

def main():
    Publication.set_basics()
    fig, axes = plt.subplots(1, 2)
    axes = axes.flatten()
    plot_ramzi(axes)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main2()
