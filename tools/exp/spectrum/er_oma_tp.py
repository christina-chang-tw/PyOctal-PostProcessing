"""
er_oma_tp.py

This script is used to calculate the ER, OMA and TP of a ring resonator for a given biased and amplitude.
This can be used to investigate the performance of the ring resonator for different biasing conditions and 
find the optimal operating wavelength.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.utils.formatter import Publication
from postprocessing.parser import Parser
from postprocessing.analysis import PAnalysis
from postprocessing.utils.conversion import db2w

def main():
    Publication.set_basics()

    folder = Path(r"/path/to/folder")
    
    voltages = np.arange(1, 4.5, 0.5)
    amplitude = 0.5
    target_wavelength = 1550E-09
    input_power = 0 # dbm
    
    for volt in voltages:
        fig = plt.figure()
        df = Parser.omr_parse(folder / f"ring{volt}V.omr")
        wavelength = df["Wavelength"]
        
        # assume that all files have the same wavelength points
        losses_1 = Parser.omr_parse(folder / f"ring{volt + amplitude:.2f}V.omr", convert_to_csv=True)["Loss [dB]"]
        losses_2 = Parser.omr_parse(folder / f"ring{volt - amplitude:.2f}V.omr", convert_to_csv=True)["Loss [dB]"]
        
        # operating region
        er_spectrum = PAnalysis.er(wavelength, losses_1, losses_2, target_wavelength)
        oma_spectrum = PAnalysis.oma(wavelength, losses_1, losses_2, target_wavelength)
        tp_spectrum = np.absolute(10*np.log10(2*db2w(input_power)/oma_spectrum)) # db to W == dbm to mW

        fig.plot(wavelength, df["Loss [dB]"], label="ILoss")
        fig.plot(wavelength, oma_spectrum, label="OMA")
        fig.plot(wavelength, er_spectrum, label="ER")
        fig.plot(wavelength, tp_spectrum, label="TP")
    
    plt.show()
    
if __name__ == "__main__":
    main()
    
    
        
        
        