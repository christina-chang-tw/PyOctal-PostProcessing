from pathlib import Path

import pandas as pd
import numpy as np
import re

from postprocessing.iomr import IOMRFileHandler, IOMRGraphHandler

class Parser:

    @staticmethod
    def __convert_freq(data):
        if "GHz" in data:
            return float(data.strip(" GHz"))*1e9
        elif "MHz" in data:
            return float(data.strip(" MHz"))*1e6
        elif "kHz" in data:
            return float(data.strip(" kHz"))*1e3

    @staticmethod
    def ads_parse(filename: Path):
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()[:2]
        ind_var = lines[0].strip("\n\"")
        dep_var = re.findall(r"'(.*?)'", lines[1])
        names = [ind_var] + dep_var

        content = pd.read_csv(filename, skiprows=1, header=None, delimiter=",", names=names)
        if "freq" in names:
            content["freq"] = content["freq"].apply(lambda x: Parser.__convert_freq(x))

        content = content.dropna(subset=["freq"]).reset_index(drop=True)
        content = content.apply(lambda x: pd.to_numeric(x, errors="ignore"), axis=1)
        return content

    @staticmethod
    def vcsv_parse(filename: Path):
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()

        content = dict()
        labels = lines[1].lstrip(";").rstrip("\n").split(",;")
        print(labels)
        df = pd.DataFrame([data.split(",") for data in lines[6:]], dtype=float)
        for idx, label in enumerate(labels):
            content[label] = np.round((df[idx*2], df[idx*2+1]), 9)

        return content
    
    @staticmethod
    def snp_parse(filename: Path, fmt: str="dB"):
        with open(filename, "r", encoding="utf-8") as file:
            raw_data = file.readlines()
        port = int(filename.suffix[2])

        data = [line.split() for line in raw_data if line[0].isnumeric()]
        df = pd.DataFrame(data, dtype=float)
        df["freq"] = df[0]
        if fmt.upper() == "RI": # real and imaginary
            sparams = [f"s{j}{i}" for i in range(1, port + 1) for j in range(1, port + 1)]
            for i, sparam in zip(range(1, len(sparams)*2, 2), sparams):
                df[sparam] = df[[i, i+1]].apply(lambda n: n.iloc[0]+1j*n.iloc[1], axis=1)

        # angle in deg
        else: # magnitude angle, dB angle
            sparams = []
            for i in range(1, port + 1):
                for j in range(1, port + 1):
                    sparams.extend([f's{j}{i}', f's{j}{i} angle'])
            for i, sparam in enumerate(sparams, start=1):
                df[sparam] = df[i]

        df.drop(columns=[i for i in df.columns if isinstance(i, int)], inplace=True)

        return df

    def iomr_parse(filename: Path, meas: str="RXTXAvgIL"):
        # Currently only implement one method

        omrfile = IOMRFileHandler()
        filepath = filename.resolve()

        omrfile.read_omr(filepath)
        graph = IOMRGraphHandler(omrfile.graph(meas))

        return pd.DataFrame({"Wavelength": graph.xdata, "Loss [dB]": graph.ydata})
