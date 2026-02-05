import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

class MZI():
    def __init__(xdata: np.ndarray, ydata: np.ndarray, target_wavelength: float, cutoff: float = 10):
        """_summary_

        Args:
            xdata (np.ndarray): _description_
            ydata (np.ndarray): should be in decibels (dB) scale
            target_wavelength (float): _description_
            cutoff (float, optional): _description_. Defaults to 10.
        """
        self.xdata = xdata
        self.ydata = ydata
        self.target_wavelength = target_wavelength
        self.cutoff = cutoff

def main2():
    file = Path(r"C:\Users\tyc1g20\Downloads\results_dim_r0um_20250821_114310.csv")
    df = pd.read_csv(file)

    fig, ax = plt.subplots(figsize=(5, 4))
    lengths = np.linspace(20E-06, 500E-06)
    voltages = np.arange(0, 7, 1)

    df["neff"] = df["neff"].apply(lambda x: complex(x.replace("i", "j"))).values
    # dneff = np.real(neff - neff[0])
    # length = 1E-3
    # phase_change = np.absolute(360 * dneff * length / 1.55E-06)
    # lengths = (90 / phase_change) * (1 / length)
    
    for v in voltages:
        loss = df[df["voltages (V)"] == v]["loss (dB/cm)"].values
        dneff = df[df["voltages (V)"] == v]["neff"].values - df[df["voltages (V)"] == 0]["neff"].values
        phase_change = np.absolute(360 * dneff / 1.55E-06)
        length = (90 / phase_change)

        # ax.scatter(length * 1E6, loss * length * 1E2, marker='x', s=50)
        print(length * 1E6, loss * length * 1E2)
        ax.plot(lengths * 1E6, loss * lengths * 1E2, label=f"{v}V")
    ax.set_xlabel("Length (um)")
    ax.set_ylabel("Insertion loss (dB)")
    ax.set_ylim(0, 10)
    ax.legend()

    ax.grid(True, which='both', linestyle='--', linewidth=0.8)
    ax.set_xlim(np.min(lengths) * 1E6, np.max(lengths) * 1E6)

    ax.set_title("tox=5nm, hdp=0.6um")
    fig.tight_layout()
    fig.savefig("figure.png", dpi=400, bbox_inches='tight')
    plt.show()


def main():
    lengths = [80, 100, 125]
    tox = 3
    folder = Path(r"C:\Users\tyc1g20\Downloads\mzis")
    voltages = np.arange(0, 6.5, 0.5)

    fig, ax = plt.subplots(figsize=(6, 4))

    for length in lengths:
        iloss = []
        file = folder / f"mzi_tox{tox}nm_hdp0.2um_l{length}um.csv"
        df = pd.read_csv(file)
        for v in voltages:
            iloss.append(np.max(df[f"{v:g}V"]))
        ax.plot(voltages, iloss, label=f"{tox}nm, hdp0.2um, {length}um", marker='o')

    fig.savefig(f"figure.png", dpi=400, bbox_inches='tight')

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Insertion loss (dB)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.8)  

    ax.set_xlim(0, 6)
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main2()