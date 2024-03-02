from os import path, listdir
from sys import argv
from pathlib import Path
import argparse

import pandas as pd
from tqdm import tqdm

from postprocessing.utils.formatter import CustomArgparseFormatter
from postprocessing.iomr import IOMRFileHandler, IOMRGraphHandler

def parse(args: str=None) -> dict[str,str]:
    parser = argparse.ArgumentParser(
        description="Convert .omr files to .csv files.",
        formatter_class=CustomArgparseFormatter)
    parser.add_argument(
        "--input-dir",
        metavar="",
        nargs=1,
        type=str,
        help="The directory containing the .omr files.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        metavar="",
        nargs=1,
        type=str,
        help="The directory where the .csv files will be saved.",
        required=True,
    )
    parser.add_argument(
        "--measurement",
        dest="meas",
        metavar="",
        nargs=1,
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
    
    files = listdir(args.input_dir)
    
    omrfile = IOMRFileHandler()
    for file in tqdm(files, total=len(files), desc="Converting files"):
        if file.endswith('.omr'):
            filepath = path.join(args.input_dir, file)

            omrfile.read_omr(filepath)
            graph = IOMRGraphHandler(omrfile.graph(args.meas))

            output_filepath = path.join(args.output_dir, f"{Path(file).stem}.csv")
            pd.DataFrame({"Wavelength": graph.xdata, "Loss [dB]": graph.ydata}).to_csv(output_filepath, index=False)

if __name__ == "__main__":
    convert(argv[1:])