from postprocessing.parser import Parser
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_against_voltage(axis, axis2, data):
    filters = np.append(np.logspace(8, 10, 3), [60e+09, 39810717055.34969, 3981071705.5349693])
    axis.scatter(data["Vg"], data["C_dc"]*1E+9, label="C_dc", s=15)
    for fidx, idx in enumerate(range(len(data["f"]))):
        if data["f"][fidx] in filters:
            axis.scatter(data["Vg"], data["C_ac"][idx]/data["leng"]*1E+9,label=data["f"][fidx]/1E+9)
            z_unzipped = list(zip(*data["Z"][idx]))
            axis2.scatter(data["Vg"], np.array(z_unzipped[0]), label=data["f"][fidx]/1E+9)
            print("f: ", data["f"][fidx]/1E+9, "C: ", data["C_ac"][idx]/data["leng"]*1E+9)

        
    axis.legend()
    axis.set_xlabel("Voltage [V]")
    axis.set_ylabel("Capacitance [fF/um]")
    axis2.legend()
    axis2.set_xlabel("Voltage [V]")
    axis2.set_ylabel("Resistance [Ohm]")
    axis2.legend()

def plot_against_frequency(axis, axis2, data):
    vgate = sorted(np.abs(data["Vg"]))
    filters = np.arange(0, 7, 1)

    for idx, v in enumerate(vgate):
        if v in filters:
            axis.plot(data["f"], data["C_ac"][:,idx], label=f"{v}V")
            z = data["Z"][:,idx]
            unzipped = list(zip(*z))

            axis2.plot(data["f"], unzipped[0], label=f"{v}V")
            

    axis.legend()
    axis.set_xscale("log")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Capacitance [fF/um]")
    axis2.legend()
    axis2.set_xscale("log")
    axis2.set_xlabel("Frequency [Hz]")
    axis2.set_ylabel("Resistance [Ohm]")
    axis2.legend()

def main():
    fig, ax = plt.subplots(1,2)
    ax = ax.flatten()

    filename = Path(r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\3nm_ac_data_3.mat")
    data = Parser.matlab_parse(filename)
    # set the dot size
    plot_against_frequency(ax[0], ax[1], data)
    

    

    plt.show()

if __name__ == "__main__":
    main()