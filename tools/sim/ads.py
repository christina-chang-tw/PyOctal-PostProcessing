"""
ads.py

This script is used to plot the effective capacitance and resistance of a device for different biasing conditions.
The file should be a csv exported from ADS Keysight.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.parser import Parser
from postprocessing.utils.formatter import Publication

def main():
    Publication.set_basics()
    
    folder = Path(r"path/to/folder")
    variables = np.arange(1, 5)
    target_freq = 40E+09
    
    ceff, reff = [], []
    
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    for var in variables:
        data = Parser.ads_parse(folder / f"{var}.csv")
        
        idx = np.where(data["freq"] == target_freq)[0][0]
        ceff.append(data["Ceff"][idx])
        reff.append(data["Reff"][idx])
        
        ax[0].plot(data["freq"]/1E+09, np.real(data["Reff"]), label=f"{var}V")
        ax[1].plot(data["freq"]/1E+09, data["Ceff"]*1E+15, label=f"{var}V")
        
    ax[0].set_xlabel("Frequency [GHz]")
    ax[0].set_ylabel(r"Re{Reff} [$\Omega$]")
    ax[0].set_xscale("log")
    ax[0].legend()
    
    ax[1].set_xlabel("Frequency [GHz]")
    ax[1].set_ylabel("Ceff [fF]")
    ax[1].set_xscale("log")
    ax[1].legend()
    
    ax[2].scatter(variables, np.real(np.array(reff)), marker=".")
    ax[2].set_xlabel("Voltage [V]")
    ax[2].set_ylabel(r"Re{Reff}" + f" @ {target_freq/1E+9}GHz" + r"[$\Omega$]")
    
    ax[3].scatter(variables, np.array(ceff)*1E+15, marker=".")
    ax[3].set_xlabel("Voltage [V]")
    ax[3].set_ylabel(f"Ceff @ {target_freq/1E+9}GHz [fF]")
    
    
    
        
    
        
        
        