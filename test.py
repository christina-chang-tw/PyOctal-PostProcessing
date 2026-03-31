import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from postprocessing.utils.formatter import Publication
from postprocessing.analysis import PAnalysis

def plot_loss(axis):
    prefix = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\MOS_EAM_S"
    thicknesses = range(2, 6)
    axis2 = axis.twinx()

    fsr = 7.1e-09
    
    linestyles = ["solid", "dashed", "dashdot", "dotted"]

    for idx, thickness in enumerate(thicknesses):
        csv_fname=f'{prefix}{thickness}nm_10_11_2023.txt'
        df = np.loadtxt(csv_fname)
        _, loss = PAnalysis.get_loss(1550e-09, df[:,0], df[:,2])
        axis.plot(np.absolute(df[:,0]), np.absolute(loss)/10, label=f"{thickness}nm", linestyle=linestyles[idx], color="C0")
        print("thickness: ", thickness, "loss: ", np.abs(loss[-1])/10, "voltage: ", df[-1,0])
    
    filepath = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\r15um_lambda_shift.csv"

    df = pd.read_csv(filepath)
    fsr = 7.1e-09

    voltage = df.pop("Voltage (V)")
    
    for idx, (key, val) in enumerate(df.items()):
        axis2.plot(np.absolute(voltage), 2*val/fsr, label=key, linestyle=linestyles[idx], color="C1")
    
    axis2.set_xlabel("Voltage [V]")
    axis2.set_ylabel(r"Phase change [$\pi$]")
    axis2.set_ylim([-0.2, 0])
    # set y axis color
    axis2.spines['left'].set_color('C0')
    axis2.spines['right'].set_color('C1')
    axis.spines['left'].set_color('C0')
    axis.spines['right'].set_color('C1')
    # set y axis label color
    axis2.yaxis.label.set_color('C1')
    axis.yaxis.label.set_color('C0')
    # set y axis ticks color
    axis2.tick_params(axis='y', colors='C1')
    axis.tick_params(axis='y', colors='C0')
    
    axis.set_xlabel("Voltages [V]")
    axis.set_ylabel(r'|$\Delta\alpha$| [dB/mm]')
    axis.set_ylim([0, 20])
    axis.set_xlim([0, 6])
    axis.set_title("(b)", fontsize=18)
    axis.legend(loc=[0,0.3])

    # axis2.set_ylim([0, ])
    axis.set_yticks(np.linspace(axis.get_ybound()[0], axis.get_ybound()[1], 6))
    axis2.set_yticks(np.linspace(axis2.get_ybound()[0], axis2.get_ybound()[1], 6))

def plot_modeff(axis):
    prefix = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\MOS_EAM_S"
    thicknesses = range(2, 6)


    for idx, thickness in enumerate(thicknesses):
        csv_fname=f'{prefix}{thickness}nm_10_11_2023.txt'
        df = np.loadtxt(csv_fname)
        v, eff = PAnalysis.get_modeff(wavelength=1550.36e-09, voltages=np.absolute(df[:,0]), dneff=df[:,1])
        axis.plot(np.absolute(v), np.absolute(eff*1e03), label=f"{thickness}nm")
        print("thickness: ", thickness, "eff: ", eff[-1]*1e03, "voltage: ", v[-1])

    axis.set_title("(c)", fontsize=18)
    axis.set_xlabel("Voltages [V]")
    axis.set_ylabel("Modulation Efficiency [Vmm]")
    axis.set_xlim([0, 6])
    axis.set_ylim([0, 25])
    
    axis.legend()


def plot_moscaps_oma(axis):
    thicknesses = range(2, 6)
    prefix = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\post-processing\\"
    for thickness in thicknesses:
        filename = f"{prefix}MOS_EAM_S{thickness}nm_fom_1v_3v_13_11_2023.csv"
        data = pd.read_csv(filename)
        axis.plot(data["Wavelength"]*1e+09, data["Optical modulation amplitude [W]"], label=f"{thickness}nm")

    axis.set_xlabel("Translated Wavelength [nm]")
    axis.set_ylabel("Normalised OMA")
    axis.set_xlim([-0.5, 0.5])
    axis.set_ylim([0, 0.85])
    axis.set_title("(a)", fontsize=18)
    axis.legend()


def plot_total_capacitance(ax):

    ax2 = ax.twinx()
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    for idx, thickness in enumerate([2,3,4,5]):

        # find files in the current directory with the extension .txt but not containing a substring pos
        cap_file = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\cap files\moscap_n6v_cap_" + str(thickness) + "nm.txt"
        mod_eff =  r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\MOS_EAM_S" + str(thickness) + "nm_10_11_2023.txt"

        # Load the data
        cap_data = np.loadtxt(cap_file)
        mod_eff_data = np.loadtxt(mod_eff)
        veff, modeff = PAnalysis.get_modeff(1550e-09, mod_eff_data[:,0], mod_eff_data[:,1])
        voltage, total_cap = PAnalysis.total_capacitance(veff=veff, vcap=cap_data[:,0], eff=modeff, cap=cap_data[:,1])

        
        ax.plot(np.absolute(cap_data[:,0]), np.absolute(cap_data[:,1])*10**9, label=f"{thickness} nm", linestyle=linestyles[idx], color="C0")
        ax2.plot(np.absolute(voltage), np.absolute(total_cap)*10**12, label=f"{thickness} nm", linestyle=linestyles[idx], color="C1")

    ax2.set_xlabel("Voltage [V]")
    ax2.set_ylabel(r"$C_T$ [pF]")
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel(r"$C'$ [fF/um]")

    ax.set_xlim([0, 6])
    ax.set_title("(d)", fontsize=18)

    # set y axis color
    ax2.spines['left'].set_color('C0')
    ax2.spines['right'].set_color('C1')
    ax.spines['left'].set_color('C0')
    ax.spines['right'].set_color('C1')
    # set y axis label color
    ax2.yaxis.label.set_color('C1')
    ax.yaxis.label.set_color('C0')
    # set y axis ticks color
    ax2.tick_params(axis='y', colors='C1')
    ax.tick_params(axis='y', colors='C0')
    ax.legend()
    


def main():
    Publication.set_basics()
    fig, ax = plt.subplots(2, 2, figsize=(8, 7))

    ax = ax.flatten()
    plot_moscaps_oma(ax[0])
    plot_loss(ax[1])
    plot_modeff(ax[2])
    plot_total_capacitance(ax[3])
    fig.tight_layout()
    fig.savefig("capacitance.pdf")
    plt.show()

    

if __name__ == "__main__":
    main()