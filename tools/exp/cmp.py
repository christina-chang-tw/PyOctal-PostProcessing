from pathlib import Path
from typing import List
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from postprocessing.parser import Parser
from postprocessing.utils.formatter import Publication


def wafer_plots():
    folder = Path(r"C:\Users\Christine\Downloads\Polishing")
    files = ["Slot14_Position_CMP1.txt", "Slot14_Position_CMP1.txt"] # in ascending order

    for f in files:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        file = folder / f
        df = Parser.wafer_txt_parse(file)
        df = df.dropna().reset_index()

        contour = ax.tricontourf(df["X"], df["Y"], df["Z"], levels=30, cmap='jet')
        ax.scatter(df["X"], df["Y"], c="black", s=12, alpha=0.6)
        fig.colorbar(contour, ax=ax, shrink=0.7, label='Rate (nm/min)')

        ax = Publication.add_wafer_circles(ax)

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(file.with_suffix(".png"), dpi=400)



def wafer_polish_rate():
    folder = Path(r"C:\Users\Christine\Downloads\Polishing")
    files = ["slot1_10_min.txt", "slot1_15_min.txt", "slot1_21_min.txt"] # in ascending order
    polish_time = [5, 6]
    # zlim = [200, 100]
    zlim = [None, 300]

    total_ptime = 10
    for i, (ptime)in enumerate(polish_time):
        total_ptime += ptime
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        df_sio2_prev = Parser.wafer_txt_parse(folder / files[i])
        df_sio2_curr = Parser.wafer_txt_parse(folder / files[i+1])
        merged = pd.merge(df_sio2_prev, df_sio2_curr, on=["X", "Y"], how="inner", suffixes=("_prev", "_curr"))

        merged['Z_diff'] = merged['Z_prev'] - merged['Z_curr']
        merged = merged[(merged["Z_diff"] > 0)]
        if len(zlim) and zlim[i] is not None:
            merged = merged[(merged["Z_diff"] < zlim[i])]
        merged['rate (nm/min)'] = merged['Z_diff'] / ptime
        merged = merged.dropna()

        contour = ax[0].tricontourf(merged["X"], merged["Y"], merged["Z_diff"], levels=30, cmap='jet')
        ax[0].scatter( merged["X"], merged["Y"], c="black", s=12, alpha=0.6, label="Sample Points")
        fig.colorbar(contour, ax=ax[0], shrink=0.7, label='Thickness (um)')

        contour = ax[1].tricontourf(merged["X"], merged["Y"], merged["rate (nm/min)"], levels=30, cmap='jet')
        ax[1].scatter(merged["X"], merged["Y"], c="black", s=12, alpha=0.6)
        fig.colorbar(contour, ax=ax[1], shrink=0.7, label='Rate (nm/min)')

        ax[0] = Publication.add_wafer_circles(ax[0])
        ax[1] = Publication.add_wafer_circles(ax[1])

        ax[0].set_xlabel("X (mm)")
        ax[0].set_ylabel("Y (mm)")
        ax[0].set_aspect('equal')
        avg_diff = np.average(merged['Z_diff'])
        med_diff = np.median(merged['Z_diff'])
        ax[0].set_title(
            f'Polished thickness period: {total_ptime-ptime:g} min. - {total_ptime:g} min. \n'
            f'Avg: {avg_diff:.2f} nm, Median: {med_diff:.2f} nm'
        )
        ax[0].set_xlim(-10, 10)
        ax[0].set_ylim(-10, 10)

        ax[1].set_xlabel("X (mm)")
        ax[1].set_ylabel("Y (mm)")
        ax[1].set_aspect('equal')
        ax[1].set_title(
            f'Polishing rate period: {total_ptime-ptime:g} min. - {total_ptime:g} min. \n'
            f'Avg: {np.average(merged["rate (nm/min)"]):.2f} nm/min, Median: {np.median(merged["rate (nm/min)"]):.2f} nm/min'
        )
        ax[1].set_xlim(-10, 10)
        ax[1].set_ylim(-10, 10)
        fig.tight_layout()
        fig.savefig(f"polish_time_delta_{total_ptime}", dpi=400)
        plt.show()


def wafer_underlayer_polish_rate():
    folder = Path(r"C:\Users\Christine\Downloads\Polishing")
    files_Si = ["Slot14_CMP3.txt", "Slot14_CMP0.txt", "Slot14_CMP1.txt", "Slot14_CMP2.txt", "Slot14_CMP3.txt"]
    files_SiO2 = ["Slot8_21_min_SiO2.txt", "Slot8_25_min_SiO2.txt", "Slot8_27_min_SiO2.txt", "Slot8_27_min_SiO2.txt"] # in ascending order
    polish_time = [12, 12, 12]

    total_ptime = 21
    for i, ptime in enumerate(polish_time):
        total_ptime += ptime
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # target and baseline
        df_base = Parser.wafer_txt_parse(folder / files_Si[i]).rename(columns={"Z": "Z_base"})
        df_tar_prev = Parser.wafer_txt_parse(folder / files_SiO2[i])
        df_tar_curr = Parser.wafer_txt_parse(folder / files_SiO2[i+1])
        merged = pd.merge(df_tar_prev, df_tar_curr, on=["X", "Y"], how="inner", suffixes=("_prev", "_curr"))
        merged = pd.merge(merged, df_base, on=["X", "Y"], how="inner")

        merged['Z_diff'] = merged['Z_prev'] - merged['Z_curr']
        merged['rate (nm/min)'] = merged['Z_diff'] / ptime
        merged = merged[(merged["Z_base"] < 50) & (merged["Z_diff"] > 0)]
        merged = merged.dropna()

        # Plot sample point locations
        ax[0].scatter(merged["X"], merged["Y"], c="black", s=10, alpha=0.5)
        ax[1].scatter(merged["X"], merged["Y"], c="black", s=10, alpha=0.5)

        # contour = ax[0].tricontourf(merged["X"], merged["Y"], merged["Z_diff"], levels=30, cmap='jet')
        # ax[0].scatter( merged["X"], merged["Y"], c="black", s=12, alpha=0.6, label="Sample Points")
        # fig.colorbar(contour, ax=ax[0], shrink=0.7, label='Thickness (um)')

        # contour = ax[1].tricontourf(merged["X"], merged["Y"], merged["rate (nm/min)"], levels=30, cmap='jet')
        # ax[1].scatter(merged["X"], merged["Y"], c="black", s=12, alpha=0.6)
        # fig.colorbar(contour, ax=ax[1], shrink=0.7, label='Rate (nm/min)')

        merged.to_csv(f"slot8_{total_ptime}_minutes_overpolish.csv", index=False)

        for _, row in merged.iterrows():
            # Thickness values on left subplot (ax[0])
            ax[0].annotate(
                f"{row['Z_diff']:.0f}", 
                (row["X"], row["Y"]), 
                fontsize=10, 
                ha='center', 
                va='bottom',
                xytext=(0, 2), 
                textcoords='offset points'
            )
            
            # Rate values on right subplot (ax[1])
            ax[1].annotate(
                f"{row['rate (nm/min)']:.0f}", 
                (row["X"], row["Y"]), 
                fontsize=10, 
                ha='center', 
                va='bottom',
                xytext=(0, 2), 
                textcoords='offset points'
            )

        ax[0] = Publication.add_wafer_circles(ax[0])
        ax[1] = Publication.add_wafer_circles(ax[1])

        ax[0].set_xlabel("X (mm)")
        ax[0].set_ylabel("Y (mm)")
        ax[0].set_aspect('equal')
        avg_diff = np.average(merged['Z_diff'])
        med_diff = np.median(merged['Z_diff'])
        ax[0].set_title(
            f'Polished thickness period: {total_ptime-ptime:g} min. - {total_ptime:g} min. \n'
            f'Avg: {avg_diff:.2f} nm, Median: {med_diff:.2f} nm'
        )
        ax[0].set_xlim(-10, 10)
        ax[0].set_ylim(-10, 10)

        ax[1].set_xlabel("X (mm)")
        ax[1].set_ylabel("Y (mm)")
        ax[1].set_aspect('equal')
        avg_rate = np.average(merged["rate (nm/min)"])
        med_rate = np.median(merged["rate (nm/min)"])
        ax[1].set_title(
            f'Polishing rate period: {total_ptime-ptime:g} min. - {total_ptime:g} min. \n'
            f'Avg: {avg_rate:.2f} nm/min, Median: {med_rate:.2f} nm/min'
        )
        ax[1].set_xlim(-10, 10)
        ax[1].set_ylim(-10, 10)
        
        fig.tight_layout()
        fig.savefig(f"polish_time_delta_{total_ptime}", dpi=400)
        plt.show()



def main():
    # wafer_plots()
    # wafer_polish_rate()
    # wafer_underlayer_polish_rate()


    folder = Path(r"C:\Users\Christine\Downloads\Polishing")
    df = Parser.wafer_txt_parse(folder / "slot1_30_min.txt")
    df = df[(df["Z"] < 150)]
    print(df.head())
    df.to_csv(f"stylus.csv", index=False)


if __name__ == "__main__":
    main()

