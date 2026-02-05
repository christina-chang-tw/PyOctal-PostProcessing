from pathlib import Path
import re
from typing import Dict

import pandas as pd
import numpy as np
import win32com.client
import scipy.io as sio
import h5py

class IOMRFileHandler:
    """
    Optical Measurement Result (OMR) File Handler
    """
    def __init__(self):
        self._engine = win32com.client.Dispatch('AgServerOMRFileHandler.OMRFile')

    @property
    def engine(self):
        return self._engine
    
    @engine.setter
    def engine(self, engine):
        self._engine = engine

    @property
    def figure(self):
        return self._figure
    
    @figure.setter
    def figure(self, figure):
        self._figure = figure

    # IOMR File Handler
    def properties(self, name: str):
        return self.engine.Property(name)
    
    def graph_names(self):
        return self.engine.GraphNames
    
    def graph(self, name: str):
        return self.engine.Graph(name)
    
    def plugins(self):
        return self.engine.Plugins
    
    def plugin(self, name: str):
        self._plugin = self.engine.Plugin(name)
        return self._plugin
    
    def write_omr(self, filename: str):
        self.engine.Write(filename)

    def read_omr(self, filename: str):
        self.engine.OpenRead(filename)

    def close(self):
        self.engine.Close()

    # IOMR Property Handler
    def is_file_compatible(self, graph: str):
        return self._plugin.IsFileCompatible(graph)
    
    def plugin_evaluate(self, name: str):
        return self._plugin.Evaluate(name)
    
    @property
    def settings_xml(self):
        return self._plugin.SettingsXML
    
    @settings_xml.setter
    def settings_xml(self, value: str):
        self._plugin.SettingsXML = value


class IOMRGraphHandler:
    # IOMR Graph Handler
    def __init__(self, graph):
        self.graph = graph

    def properties(self, name: str):
        return self.graph.Property(name)

    def process_xdata(self):
        xdata = self.graph.XData
        num = len(self.ydata)
        if not xdata:
            xdata = np.round(np.linspace(self.xstart, self.xstop, num), 14)
        return xdata

    @property
    def xdata(self):
        return self.process_xdata()

    @property
    def ydata(self):
        return self.graph.YData

    @property
    def xstart(self):
        return self.graph.xStart

    @property
    def xstop(self):
        return self.graph.xStop

    @property
    def xstep(self):
        return self.graph.xStep

    @property
    def noChannels(self):
        return self.graph.noChannels

    @property
    def noCurves(self):
        return self.graph.noCurves

    @property
    def dataPerCurve(self):
        return self.graph.dataPerCurve


class IOMRPropertyHandler:
    def __init__(self, pty):
        self._property = pty

    # IOMR Property Handler
    def property_names(self):
        return self._property.PropertyNames

    def properties(self, name: str):
        return self._property.Property(name)

    @property
    def value(self, name: str):
        return self._property[name].Value

    @value.setter
    def value(self, name: str, value: float):
        self._property[name].Value = value

    @property
    def flag_info_pane(self, name: str):
        return self._property[name].FlagInfoPane

    @flag_info_pane.setter
    def flag_info_pane(self, name: str, value: bool):
        self._property[name].FlagInfoPane = value

    @property
    def flag_hide(self, name: str):
        return self._property[name].FlagHide

    @flag_hide.setter
    def flag_hide(self, name: str, value: bool):
        self._property[name].FlagHide = value


class Parser:
    """
    A common interface to parse difference files into a pandas dataframe format.
    """
    @staticmethod
    def __convert_freq(data):
        if "GHz" in data:
            return float(data.strip(" GHz"))*1e9
        elif "MHz" in data:
            return float(data.strip(" MHz"))*1e6
        elif "kHz" in data:
            return float(data.strip(" kHz"))*1e3

    @staticmethod
    def ads_parse(filename: Path) -> pd.DataFrame:
        """
        Keysight ADS file parser.
        
        Parameters:
            filename (Path): The path to the .csv file exported from ADS
        """
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
    def vcsv_parse(filename: Path) -> pd.DataFrame:
        """
        Cadence virtuoso csv parser.
        
        Parameters:
            filename (Path): The path to the csv file.
        """
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
    def snp_parse(filename: Path, fmt: str="dB") -> pd.DataFrame:
        """
        Keysight S-parameter parser.
        
        Parameters:
            filename (Path): The path to the s-parameter file.
            fmt (str): The format of the s-parameter file. Default is "dB".
        """
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
    
    @staticmethod
    def matlab_parse(filename: Path) -> Dict:
        """
        Matlab data parser.
        
        Parameters:
            filename (Path): The path to the .mat file.
        """
        try:
            data = sio.loadmat(filename)
            return data
        except NotImplementedError:
            file = h5py.File(filename,'r')
            def h5py_dataset_iterator(g, prefix=''):
                for key, item in g.items():
                    path = f'{prefix}.{key}' if prefix else key
                    if isinstance(item, h5py.Dataset):
                        item = np.squeeze(item)
                        yield (path, item)
                    elif isinstance(item, h5py.Group):
                        yield from h5py_dataset_iterator(item, path)
            
            data = {}
            for path, dset in h5py_dataset_iterator(file):
                data[path] = dset[()]

            return data

    def omr_parse(filename: Path, meas: str="RXTXAvgIL", convert_to_csv: bool=False) -> pd.DataFrame:
        """
        Keysight PAS file parser:
        
        Parameters:
            filename (Path): The path to the .pas file.
            meas (str): The measurement type. Default is "RXTXAvgIL".
        """
        # Currently only implement one method

        # check if the file with a suffix of .csv exists
        csv_filename = filename.with_suffix(".csv")
        if csv_filename.exists():
            return pd.read_csv(csv_filename)

        omrfile = IOMRFileHandler()
        filepath = filename.resolve()
        omrfile.read_omr(filepath)
        graph = IOMRGraphHandler(omrfile.graph(meas))

        df = pd.DataFrame({"Wavelength": graph.xdata, "Loss [dB]": graph.ydata})
        # speed up the process by converting to csv
        if convert_to_csv:
            df.to_csv(csv_filename, index=False)
        return df