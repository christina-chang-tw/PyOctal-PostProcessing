from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from postprocessing.analysis import PAnalysis
from postprocessing.fom import fom


def main():
    folder = Path(r"C:\Users\tyc1g20\Downloads")
    file = folder / f"accum_h325nm_p4E18_n4E18_tox3nm_r25um_hdp0.4um_c0.58_r0.9.csv"

    # folder = Path(r"C:\Users\tyc1g20\Downloads\41566_2023_1159_MOESM14_ESM")
    # file = "combined.csv"

    df = pd.read_csv(file)
    print(df.keys())
    wavelength = df["Wavelength (nm)"].values * 1E-9

    vstart = 1
    vstop = 5
    vstep = 1
    vpp = 1
    voltages = np.arange(vstart, vstop+vstep, vstep)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(6, 4))
    ax2 = ax.twinx()

    for volt, color in zip(voltages, colors):
        analysis = PAnalysis(wavelength, df[f"{volt:g}V"].values, 1.55E-06, cutoff=10)
        # analysis.sanity_check()

        insertion_loss = np.absolute(df[f"{volt-1:g}V"].values - np.max(df[f"{volt-1:g}V"].values))
        er = fom(ydata0=df[f"{volt-vpp/2:g}V"].values, ydata1=df[f"{volt+vpp/2:g}V"].values, fom_type="ER")

        ax2.plot(wavelength * 1E6, insertion_loss, color=color, linestyle="--")
        ax.plot(wavelength * 1E6, er, label=f"{volt}V", color=color)

        il_indices = np.where(np.isclose(er - 3, 0, atol=0.01))[0]
        er_indices = np.where(np.isclose(insertion_loss - 6, 0, atol=0.01))[0]

        print(
        f"""
        Figure of merit @ {volt}V:
            q-factor: {analysis.qfactor():.2f}
            IL @ ER=3dB: {insertion_loss[il_indices]} dB
            ER @ IL=6dB: {er[er_indices]} dB
        """)

    ax.axhline(y=3, color='C0', linestyle='-', linewidth=1.2)
    ax.axhline(y=6, color='C0', linestyle='-', linewidth=1.2)
    ax2.axhline(y=3, color='C1', linestyle='--', linewidth=1.2)
    ax2.axhline(y=6, color='C1', linestyle='--', linewidth=1.2)
    
    ax.legend()
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Extinction ratio (dB)")
    ax2.set_ylabel("Insertion loss (dB)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.8)
    # ax.set_ylim(0, 24)
    # ax2.set_ylim(0, 38)
    # ax.set_xlim(1.5495, 1.5505)
    # no e notation in the x axis
    ax.ticklabel_format(style='plain', useOffset=False)
    fig.tight_layout()

    fig.savefig("fom_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
