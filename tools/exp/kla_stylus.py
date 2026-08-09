import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import List
from scipy import signal
import numpy as np

from postprocessing.parser import Parser
from postprocessing.utils.formatter import Publication


def individual_profiles(files: List[Path], titles: List[tuple]):
    for file, pos in zip(files, titles):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        header, data = Parser.kla_stylus_parse(file)
        normal_data = data["Normal"]
        position = data["Position (um)"]
        if header.points == 1501:
            position = position + 50
        ax.plot(position, normal_data, label="Raw")
        normal_data = signal.savgol_filter(
            normal_data,
            window_length=21,
            polyorder=3,
        )
        # sos = signal.butter(10, 30, 'lp', fs=1000, output='sos')
        # normal_data = signal.sosfiltfilt(sos, normal_data)
        # normal_data = normal_data.rolling(window=50, center=True).mean()
        ax.plot(position, normal_data, label="Fitted")
        
        ax.set_title(f"({pos[0]}, {pos[1]})")
        ax.set_xlabel("Position (um)")
        ax.set_ylabel("Height (Angstroms)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(file.with_suffix(".png"), dpi=400)


def overlap_profiles(files: List[Path]):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    for file in files:
        header, data = Parser.kla_stylus_parse(file)
        normal_data = data["Normal"]
        position = data["Position (um)"]
        if header.points == 1501:
            position = position + 50
        # normal_data = signal.savgol_filter(
        #             normal_data,
        #             window_length=21,
        #             polyorder=3,
        #         )
        ax.plot(position, normal_data, label=file.stem)
        
    ax.set_title(f"Overlapped profile")
    ax.set_xlabel("Position (um)")
    ax.set_ylabel("Height (Angstroms)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(file.parent / "overlapped.png", dpi=400)


def dishing_position(files: List[Path], x: List[float], y: List[float], pos_filepath: str="position.csv"):
    filename = [f.name for f in files]
    pos = np.column_stack([filename, x, y])
    df = pd.DataFrame(pos, columns=["Filename", "x (um)", "y (um)"])
    df.to_csv(pos_filepath, index=False)

    x, y = np.array(x)/1E4, np.array(y)/1E4

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax = Publication.add_wafer_circles(ax)
    for i, file in enumerate(files):
        _, data = Parser.kla_stylus_parse(file)
        normal_data = data["Normal"]
        normal_data = signal.savgol_filter(
            normal_data,
            window_length=21,
            polyorder=3,
        )
        # normal_data = normal_data.rolling(window=50, center=True).mean()

    
        ax.scatter(x[i], y[i], c="black", s=10, alpha=0.5)
        ax.annotate(
            f"{normal_data.min():.0f}", 
            (x[i], y[i]), 
            fontsize=10, 
            ha='center', 
            va='bottom',
            xytext=(0, 1), 
            textcoords='offset points'
        )
    ax.set_title("Dishing (Angstrom)")
    ax.set_xlabel("X position (cm)")
    ax.set_ylabel("Y position (cm)")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(alpha=0.3, zorder=0)
    ax.ticklabel_format(style="plain")
    fig.tight_layout()
    fig.savefig("data.png", dpi=400)


def main():
    folder = Path(r"D:\CMP\slot14")
    # files = [
    #     Path(r"E:\CMP\slot14\after 10 mins polishing.txt"),
    #     Path(r"E:\CMP\slot14\after 24 mins polishing.txt"),
    #     Path(r"E:\CMP\slot14\after 28 mins polishing.txt"),
    #     Path(r"E:\CMP\slot14\after 32 mins polishing x0 y0.txt"),
    # ]
    files = [
        folder / "after_12_minutes.txt",
        folder / "after_12_minutes_2.txt",
        folder / "after_21_minutes.txt",
        folder / "after_21_minutes_2.txt",
        folder / "after_29_minutes_0.txt",
        folder / "after_29_minutes_1.txt",
        folder / "after_29_minutes_2.txt",
        folder / "after_29_minutes_3.txt",
        folder / "after_29_minutes_4.txt",
        folder / "after_29_minutes_5.txt",
        folder / "after_29_minutes_6.txt",
    ]

    x = [35728.77, 48528.31, 36336.53, -25216.47, -34282.83, -40151.14, 11136.3]
    y = [33097.3, 218.02, -32922.52, -21706.02, -5542.64, 32572.97, -3620.62]

    titles = list(zip(x, y))
    # individual_profiles(files=files, titles=titles)
    overlap_profiles(files=files)
    # dishing_position(files=files, x=x, y=y)
    plt.show()
    


if __name__ == "__main__":
    main()