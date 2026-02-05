from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from os.path import join

from postprocessing.utils.formatter import Publication
from postprocessing.analysis import PAnalysis


def plot_transmission_spectrum_for_different_thicknesses(axis, filepath):
    data = pd.read_csv(filepath)
    wavelength = data.pop("Wavelength")*1e+09
    voltages = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
   
    for volt in voltages:
        axis.plot(wavelength, data[str(volt)], label=f"{int(np.abs(volt))}V")
    axis.set_xlabel("Wavelength [nm]")
    axis.set_ylabel("Transmission [dB]")
    axis.legend()
    axis.set_xlim([1549.5, 1551])
    axis.set_ylim([-30, 0])
    axis.set_xticks(np.linspace(1549.5, 1551, 4))



def plot_tp_er_iloss_for_3nm(axis, filepath):
    data = pd.read_csv(filepath)
    axis.plot(data["Wavelength"]*1e+09, data["Transmission penalty[dB]"], color="C0", label="TP")
    axis.plot(data["Wavelength"]*1e+09, data["Extinction ratio [dB]"], color="C1", label="ER")
    axis.plot(data["Wavelength"]*1e+09, np.absolute(data["Insertion loss [dB]"]), color="C2", label=r"$IL_{one}$")
    axis.set_xlabel("Translated wavelength [nm]")
    axis.set_ylabel(r"ER/$IL_{one}$/TP [dB]")
    axis.set_xlim([-0.5, 0.5])
    axis.set_xticks(np.linspace(-0.4, 0.4, 5))
    axis.set_ylim([0, 35])
    axis.legend()

def plot_oma(axis, filepath, label):
    data = pd.read_csv(filepath)
    axis.plot(data["Wavelength"]*1e+09, np.absolute(data["Optical modulation amplitude [W]"]), label=label)
    axis.set_xlabel("Translated wavelength [nm]")
    axis.set_ylabel(r"OMA")
    axis.set_xlim([-0.5, 0.5])
    axis.set_xticks(np.linspace(-0.4, 0.4, 5))
    axis.set_ylim([0, 1])
    axis.legend()


def plot_linewidth(axis, axis2, filepath, label, linestyle):
    data = pd.read_csv(filepath)
    wavelength = data.pop("Wavelength")

    voltages = list(map(float,data.keys()))
    linewidths = np.zeros(shape=(len(voltages),1))
    qfactors = np.zeros(shape=(len(voltages),1))

    for idx, volt in enumerate(voltages):
        analysis = PAnalysis(wavelength, data[str(volt)], 1550e-09, cutoff=5)
        linewidths[idx] = np.absolute(analysis.linewidth())
        qfactors[idx] = np.absolute(analysis.closest_resonance()/linewidths[idx])

    linewidths = (linewidths - linewidths[0])*1E+09
    
    axis2.plot(np.absolute(voltages), linewidths, color="C1", linestyle=linestyle)
    axis.plot(np.absolute(voltages), qfactors, linestyle=linestyle, color="C0", label=label)
    axis2.set_xlabel("Voltage [V]")
    axis2.set_ylabel(r"$\Delta$ linewidth [nm]")
    axis2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.set_ylabel("Quality factor")

    # change spine color, tick color, and tick label color
    axis.spines['left'].set_color('C0')
    axis2.spines['left'].set_color('C0')
    axis.spines['right'].set_color('C1')
    axis2.spines['right'].set_color('C1')
    axis.tick_params(axis='y', colors='C0')
    axis2.tick_params(axis='y', colors='C1')
    axis2.yaxis.label.set_color('C1')
    axis.yaxis.label.set_color('C0')
    axis.legend()


def plot_oma_max(axis, folder, voltages):
    oma_max_01 = []
    oma_max_023 = []

    for vlow, vhigh in voltages:
        filepath1 = join(folder, f"MOS_EAM_t3nm_coeff_0.1_fom_v{vlow}v_{vhigh}v_13_11_2023.csv")
        filepath2 = join(folder, f"MOS_EAM_t3nm_coeff_0.23_fom_v{vlow}v_{vhigh}v_13_11_2023.csv")

        data = pd.read_csv(filepath1)
        oma_max_01.append(PAnalysis.fom_max(data["Wavelength"], np.absolute(data["Optical modulation amplitude [W]"]), "left"))

        data = pd.read_csv(filepath2)
        oma_max_023.append(PAnalysis.fom_max(data["Wavelength"], np.absolute(data["Optical modulation amplitude [W]"]), "left"))
    
    mid_voltages = np.absolute([(vl+vh)/2 for vl, vh in voltages])
    axis.plot(mid_voltages, np.array(oma_max_01), linestyle="-", color="C0", label=r"$\kappa^2$=0.1")
    axis.plot(mid_voltages, np.array(oma_max_023), linestyle="-", color="C1", label=r"$\kappa^2$=0.23")
    axis.set_xlabel("Voltage [V]")
    axis.set_ylabel("Maximum OMA")
    axis.set_xlim([1, 5])
    axis.set_ylim([0.2, 0.6])
    axis.legend()

def plot_operating_region(ax, folder, voltages):
    left_ors_01 = []
    right_ors_01 = []
    left_ors_023 = []
    right_ors_023 = []

    mid_voltages = np.absolute([(vl+vh)/2 for vl, vh in voltages])

    for vlow, vhigh in voltages:
        filepath1 = join(folder, f"MOS_EAM_t3nm_coeff_0.1_fom_v{vlow}v_{vhigh}v_13_11_2023.csv")
        filepath2 = join(folder, f"MOS_EAM_t3nm_coeff_0.23_fom_v{vlow}v_{vhigh}v_13_11_2023.csv")

        data = pd.read_csv(filepath1)
        left_or, right_or = PAnalysis.operating_region(data["Wavelength"], data["Optical modulation amplitude [W]"], 0.3)
        left_ors_01.append(left_or)
        right_ors_01.append(right_or)
    
        data = pd.read_csv(filepath2)
        left_or, right_or = PAnalysis.operating_region(data["Wavelength"], data["Optical modulation amplitude [W]"], 0.3)
        left_ors_023.append(left_or)
        right_ors_023.append(right_or)

    ax.plot(mid_voltages, np.array(left_ors_01)*1e+09, linestyle="-", color="C0", label=r"$\kappa^2$=0.1, en.")
    ax.plot(mid_voltages, np.array(right_ors_01)*1e+09, linestyle="--", color="C0", label=r"$\kappa^2$=0.1, sup.")
    ax.plot(mid_voltages, np.array(left_ors_023)*1e+09, linestyle="-", color="C1", label=r"$\kappa^2$=0.23, en.")
    ax.plot(mid_voltages, np.array(right_ors_023)*1e+09, linestyle="--", color="C1", label=r"$\kappa^2$=0.23, sup.")
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Operating region width [nm]")
    ax.set_xlim([1, 5])
    ax.set_ylim([0, 0.25])
    ax.legend(loc=[0.1,0.05])


def main2():
    Publication.set_basics()
    fig, ax = plt.subplots(2,4, figsize=(15,8))
    ax = ax.flatten()

    filepath1 = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\t3nm_coeff_0.1_ts.csv"
    filepath1_fom = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\t3nm_coeff_0.1_fom_1v_3v_13_11_2023.csv"
    plot_transmission_spectrum_for_different_thicknesses(ax[0], filepath=filepath1)
    ax[0].set_yticks(np.linspace(-20, 0, 6))
    plot_tp_er_iloss_for_3nm(ax[1], filepath=filepath1_fom)
    plot_oma(ax[2], filepath=filepath1_fom, label=r"$\kappa^2$=0.1")

    filepath2 = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\t3nm_coeff_0.23_ts.csv"
    filepath2_fom = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\t3nm_coeff_0.23_fom_1v_3v_13_11_2023.csv"
    plot_transmission_spectrum_for_different_thicknesses(ax[4], filepath=filepath2)
    ax[4].set_yticks(np.linspace(-25, 0, 6))
    plot_tp_er_iloss_for_3nm(ax[5], filepath=filepath2_fom)
    plot_oma(ax[2], filepath=filepath2_fom, label=r"$\kappa^2$=0.23")
    
    axis2 = ax[7].twinx()
    plot_linewidth(ax[7], axis2, filepath1, label=r"$\kappa^2$=0.1", linestyle="-")
    plot_linewidth(ax[7], axis2, filepath2, label=r"$\kappa^2$=0.23", linestyle="--")
    axis2.set_ylim([0, 0.25])
    ax[7].set_ylim([2500, 6500])
    ymin, ymax = ax[7].get_ybound()
    ax[7].set_yticks(np.linspace(ymin, ymax, 5))
    ymin, ymax = axis2.get_ybound()
    axis2.set_yticks(np.linspace(ymin, ymax, 5))
    ax[7].set_xticks(np.linspace(0, 6, 7))
    ax[7].legend(loc=(0.03, 0.6))

    op_folder = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\fom"
    voltages = [(v, v-2) for v in np.arange(0, -4.25, -0.25)]
    plot_operating_region(ax[6], op_folder, voltages)
    plot_oma_max(ax[3], op_folder, voltages)

    ax[0].set_title("(a)", fontsize=18)
    ax[1].set_title("(c)", fontsize=18)
    ax[2].set_title("(e)", fontsize=18)
    ax[3].set_title("(g)", fontsize=18)
    ax[4].set_title("(b)", fontsize=18)
    ax[5].set_title("(d)", fontsize=18)
    ax[6].set_title("(f)", fontsize=18)
    ax[7].set_title("(h)", fontsize=18)

    ax[2].set_ylim([0, 0.7])
    ax[0].set_ylim([-25, 0])
    
    fig.tight_layout()
    fig.savefig("fig-1-v2.pdf")
    plt.show()

if __name__ == "__main__":
    main2()