from pathlib import Path

from postprocessing.analysis import PAnalysis
import numpy as np
import matplotlib.pyplot as plt

def plot_modeff(axis):
    folder = r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\conc_sweeps\interconnect"
    files = Path(folder).rglob("*.txt")

    for file in files:
        data = np.loadtxt(file, delimiter=" ")
        _, modeff = PAnalysis.get_modeff(1550e-09, data[:, 0], data[:, 1])
        axis.scatter(np.absolute(data[:, 0]), modeff*1e+03, label=file.stem)

    data = np.loadtxt(r"\\filestore.soton.ac.uk\users\tyc1g20\mydocuments\moscap\MOS_EAM_S3nm_10_11_2023.txt")
    _, modeff = PAnalysis.get_modeff(1550e-09, data[:, 0], data[:, 1])
    axis.scatter(np.absolute(data[:, 0]), modeff*1e+03, label="3nm")

    axis.set_xlabel("Voltage [V]")
    axis.set_ylabel("Modulation efficiency [Vmm]")
    axis.legend()

if __name__ == "__main__":
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    plot_modeff(ax)
    plt.show()
