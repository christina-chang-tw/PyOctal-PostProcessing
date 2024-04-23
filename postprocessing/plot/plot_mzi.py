from os import listdir
from os.path import join
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

def db_to_linear(db: float):
    return 10**(-db/10)

def plot_mzi_from_power():
    input_dir = r"C:\Users\cchan\Desktop\PyOctal-PostProcessing\2024-4-19-ring-assisted-mzi\2024-4-19-ring-assisted-mzi-find-mixmax-g220"

    fig, ax = plt.subplots()

    files = listdir(input_dir)
    ignores = ["max_min_voltages.csv", "ring2.0_heater_2.0.csv", "ring2.5_heater_2.0.csv", "ring3.0_heater_2.0.csv"]
    files = [file for file in files if file not in ignores]

    for file in files:
        df = pd.read_csv(join(input_dir,file))
        # plt.plot(df["Voltage [V]"], df["Power [W]"])
        ax.plot(df["Voltage [V]"]*df["Current [A]"], df["Power [W]"], label=file)

    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_mzi_from_transmission_spectrum():
    input_dir = r"C:\Users\cchan\Desktop\PyOctal-PostProcessing\2024-4-19-ring-assisted-mzi\2024-4-19-ring-assisted-mzi-csv"
    
    files = listdir(input_dir)
    
    target_wavelength = 1545e-9
    target_rv = 0.0
    hvs = []
    dfs = []
    df = pd.DataFrame()

    fig, ax = plt.subplots()
    
    for file in tqdm(files):
        if file.endswith('.csv'):
            temp = pd.read_csv(f"{input_dir}/{file}")
            file = Path(file).stem
            rv, hv = np.round(np.array(re.findall(r"[-+]?\d*\.\d+|\d+", file), dtype=float), 3)

            if rv != target_rv:
                dfs.append(df)
                df = pd.DataFrame()
                hvs = []
                target_rv += 0.5

            if "Wavelength" not in df.keys():
                df["Wavelength"] = temp["Wavelength"]

            temp.drop(columns=["Wavelength"], inplace=True)
            temp = temp.rename(columns={"Loss [dB]": f"h{hv:.3f}"})
            df = pd.concat([df, temp], axis=1)
            hvs.append(hv)
    
    min = []
    for idx, df in enumerate(dfs):
        row = np.abs(df["Wavelength"]-target_wavelength).idxmin()
        loss = db_to_linear(df.iloc[row,1:].values)
        # loss = -df.iloc[row,1:].values
        power = np.array(hvs)**2
        ax.plot(power, loss, label=f"{idx*0.5}V")
        min.append(np.min(loss))
    print(min)
    ax.legend()
    plt.show()

            
if __name__ == "__main__":
    plot_mzi_from_transmission_spectrum()
