"""
iomr.py

Convert iomr files into csv files.
"""

from os import path, listdir, makedirs
from sys import argv
from pathlib import Path
import argparse

import pandas as pd
from tqdm import tqdm
import win32com.client
import numpy as np

from postprocessing.utils.formatter import CustomArgparseFormatter


class IOMRFileHandler:
    """
    IOptical Measurement Result (OMR) File Handler
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


def parse(args: str=None) -> dict[str,str]:
    parser = argparse.ArgumentParser(
        description="Convert .omr files to .csv files.",
        formatter_class=CustomArgparseFormatter)
    parser.add_argument(
        "--input",
        metavar="",
        type=str,
        help="The directory containing the .omr files.",
        required=True,
    )
    parser.add_argument(
        "--output",
        metavar="",
        type=str,
        help="The directory where the .csv files will be saved.",
        required=True,
    )
    parser.add_argument(
        "--measurement",
        dest="meas",
        metavar="",
        type=str,
        help="The measurement to be used.",
        required=False,
        default="RXTXAvgIL",
    )

    return parser.parse_args(args)

def convert(args_list: dict):
    """
    Function to convert .omr files to .csv files.

    Parameters:
    args_list (str): A list of arguments to be parsed.
    """
    args = parse(args_list)

    files = listdir(args.input)
    makedirs(args.output, exist_ok=True)

    for file in tqdm(files, total=len(files), desc="Converting files"):
        omrfile = IOMRFileHandler()
        if file.endswith('.omr'):
            filepath = path.abspath(path.join(args.input, file))

            omrfile.read_omr(filepath)
            graph = IOMRGraphHandler(omrfile.graph(args.meas))

            output_filepath = path.join(args.output, f"{Path(file).stem}.csv")
            pd.DataFrame({"Wavelength": graph.xdata, "Loss [dB]": graph.ydata}).to_csv(output_filepath, index=False)
        omrfile.close()

if __name__ == "__main__":
    convert(argv[1:])