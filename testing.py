from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def hampel(y, window=8, n_sigmas=3.0):
    med = y.rolling(window, center=True, min_periods=1).median()
    mad = (y - med).abs().rolling(window, center=True, min_periods=1).median()
    mad = mad.replace(0, np.nan).fillna(mad[mad > 0].min())
    return y.mask((y - med).abs() > n_sigmas * 1.4826 * mad, med)
    return y

def plot_width_and_overlap(width_file: Path, lumerical_file: Path,
                           voltages=(1.5, 3.0), toxes=(2, 3, 5, 10),
                           overlap_col="TM_Fraction"):
    df_w = pd.read_csv(width_file)
    df_l = pd.read_csv(lumerical_file)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    rows = [
        (df_w, "Effective width (um)", "Effective width (µm)"),
        (df_l, overlap_col,            "TM/TE mode overlap ratio"),
    ]

    markers = ["o", "^", "s", "D"]

    for row, (data, col_name, ylabel) in enumerate(rows):
        for col, v in enumerate(voltages):
            ax = axes[row, col]
            sub_v = data[data["Voltage (V)"] == v]
            for i, tox in enumerate(toxes):
                sub = sub_v[sub_v["Tox (nm)"] == tox].sort_values("Width (um)")
                if col_name != overlap_col:
                    y = hampel(sub[col_name].reset_index(drop=True))
                else:
                    y0 = sub[col_name].reset_index(drop=True)
                    y1 = sub["TE_Fraction"].reset_index(drop=True)
                    y0 = hampel(y0, window=4, n_sigmas=2)
                    y1 = hampel(y1, window=4, n_sigmas=2)
                    y = y0/y1
                ax.plot(sub["Width (um)"].values, y, marker=markers[i], ms=6,
                        color=colors[i], label=f"{tox} nm")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f"V = {v:g} V", fontsize=20)
            if row == 1:
                ax.set_xlabel("Waveguide core width (µm)", fontsize=18)
                ax.xaxis.set_ticks(np.arange(0.3, 0.47, 0.05))
                ax.xaxis.set_tick_params(labelsize=15)
            axes[row, col].yaxis.set_tick_params(labelsize=15)
        axes[row, 0].set_ylabel(ylabel, fontsize=15)

    axes[0, 0].legend(title="$t_{ox}$", frameon=True, fontsize=10, title_fontsize =12)
    axes[1, 0].set_ylim(0.9, 1.4)
    axes[1, 1].set_ylim(0.75, 1.25)
    y0, y1 = axes[1, 0].get_ylim()
    axes[1, 0].axhspan(y0, 1.0, color="tab:orange", alpha=0.06, zorder=0)
    axes[1, 0].axhspan(1.0, y1, color="tab:blue",   alpha=0.06, zorder=0)
    axes[1, 0].set_ylim(y0, y1)   # axhspan can expand the limits; pin them back
    axes[1, 0].text(0.02, 1.01, "TM-favoured", transform=axes[1, 0].get_yaxis_transform(),
        ha="left", va="bottom", fontsize=18, color="tab:blue",
        style="italic", alpha=0.8)
    axes[1, 0].text(0.02, 0.99, "TE-favoured", transform=axes[1, 0].get_yaxis_transform(),
            ha="left", va="top", fontsize=18, color="tab:orange",
            style="italic", alpha=0.8)
    
    y0, y1 = axes[1, 1].get_ylim()
    axes[1, 1].axhspan(y0, 1.0, color="tab:orange", alpha=0.06, zorder=0)
    axes[1, 1].axhspan(1.0, y1, color="tab:blue",   alpha=0.06, zorder=0)
    axes[1, 1].set_ylim(y0, y1)   # axhspan can expand the limits; pin them back

    plt.rcParams["mathtext.fontset"] = "cm"   # Computer Modern, matches LaTeX body text

    offsets = {0: -0.1, 1: -0.3}   # bottom row clears tick labels + xlabel
    for i, ax in enumerate(axes.flat):
        ax.text(0.5, offsets[i // 2], rf"$\mathrm{{({chr(97 + i)})}}$",
                transform=ax.transAxes, ha="center", va="top", fontsize=20)

    fig.tight_layout()
    
    # fig.savefig("figure.pdf", bbox_inches="tight")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25)
    fig.savefig("effective_width_mode_overlap_2x2.png",
                bbox_inches="tight")
    
plot_width_and_overlap(Path(r"C:\Users\Christine\Downloads\effective_width_99.csv"), Path(r"C:\Users\Christine\Downloads\lumerical_99.csv"))