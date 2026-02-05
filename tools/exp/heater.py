"""
heater.py

Based on the provided files, this script is used to plot the electrical power against the optical power for different biasing conditions. Allow the user to determine the power required to get the pi phase shift.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from postprocessing.utils.formatter import Publication

def main():
    Publication.set_basics()
    
    folder = Path(r"/path/to/folder")
    ext = "xlsx"
    
    variables = np.arange(0, 5)
    fig, ax = plt.subplots(1, 2)
    
    for var in variables:
        filename = folder / f"{var}V.{ext}"
        if ext == "xlsx":
            with pd.ExcelFile(filename) as f:
                data = f.parse()
        elif ext == "csv" | "txt":
            data = pd.read_csv(filename)
        
        new_volt = np.linspace(data["Voltage"].min(), data["Voltage"].max(), 5000)
        new_power = np.linspace(data["Electrical power"].min(), data["Electrical power"].max(), 5000)
        volt_fitting_func = interp1d(data["Voltage"], data[:, 1], kind="cubic", fill_value="extrapolate")
        power_fitting_func = interp1d(data["Electrical power"], data[:, 1], kind="cubic", fill_value="extrapolate")
        ax[0].plot(new_volt, volt_fitting_func(new_volt)*1E+03, label=f"{var}V")
        ax[1].plot(new_power, power_fitting_func(new_power)*1E+03, label=f"{var}V")
        
    ax[0].set_xlabel("Voltage [V]")
    ax[0].set_ylabel("Optical power [mW]")
    ax[0].legend()
    
    ax[1].set_xlabel("Electrical power [W]")
    ax[1].set_ylabel("Optical power [mW]")
    ax[1].legend()
    
    # fig.savefig("heater.png", dpi=400)
    plt.show()
            
    
    
    