import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

class DictObj:
    """ Convert a dictionary to python object """
    def __init__(self, **dictionary):
        for key, val in dictionary.items():
            if isinstance(val, dict):
                self.__dict__[key] = DictObj(**val)
            else:
                self.__dict__[key] = val

    def __getitem__(self, key):
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

    def get(self, key):
        if key in self.__dict__.keys():
            return self.__getitem__(key)
        return None
    
    def items(self):
        return self.__dict__.items()
    
def window_averaging(data: np.array, window_size: int) -> np.array:
    """
    Averaging the data.
    """
    data = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
    return data.values


def normalise(df: pd.DataFrame, freq: float, columns: list) -> pd.DataFrame:
    """
    Normalise the S-parameters with respect to the reference frequency.
    """
    columns = columns if columns else df.columns

    for column in columns:
        if "freq" in column:
            continue

        idx = np.argmin(np.absolute(df["freq"] - freq))
        ref = df[column].iloc[idx]
        df[column] = df[column] - ref

    return df

def averaging(df: pd.DataFrame, columns: list, idx: float, model: str="savgol") -> pd.DataFrame:
    """
    Averaging the S-parameters with respect to the frequency.
    """
    columns = columns if columns else df.columns
    df2 = df.copy()
    for column in columns:
        if "freq" in column:
            continue

        values = df.iloc[idx:][column]

        if model == "savgol":
            df2[f"{column}_smooth"] = np.concatenate((df[column].values[:idx], savgol_filter(values, 50, 13, mode="nearest")))
        elif model == "mean":
            df2[f"{column}_smooth"] = np.concatenate((df[column].values[:idx], values.rolling(window=200, min_periods=1).mean()))
            # df2[column] = df[column].rolling(window=200, min_periods=1).mean()
        else:
            raise ValueError("Invalid model")
    return df2


def butter_lowpass_filter(data: pd.DataFrame, columns: list, cutoff_freq: float, fs: float, order: float) -> pd.DataFrame:
    """
    Butterworth low-pass filter.

    Parameters:
        data (pd.DataFrame): The data to be filtered.
        columns (list): The columns to be filtered.
        cutoff_freq (float): The cutoff frequency.
        fs (float): The sampling frequency.
        order (int): The order of the filter.

    Returns:
        pd.DataFrame: The filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    for column in columns:
        b, a = butter(order, normal_cutoff, btype='low')
        data[column] = filtfilt(b, a, data[column])
    return data