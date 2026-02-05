import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

def window_averaging(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Averaging the data by using a rolling window.
    
    Args:
        data (np.ndarray): The data to be averaged.
        window_size (int): The size of the window.

    Returns:
        np.ndarray: The averaged data.
    """
    data = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
    return data.values


def normalise(freq: np.ndarray, values: np.ndarray, fn: float) -> np.ndarray:
    """
    Normalise the S-parameters with respect to the reference frequency (fn).
    
    Args:
        freq (np.ndarray): The frequency array.
        values (np.ndarray): The s parameter.
        fn (float): The reference frequency.
    
    Returns:
        np.ndarray: The normalised S-parameters.
    """
    idx = np.argmin(np.absolute(freq - fn))
    ref = values[idx]
    values = values - ref

    return values

def averaging(df: pd.DataFrame, columns: list, idx: int=0, model: str="savgol") -> pd.DataFrame:
    """
    Averaging the S-parameters with respect to the frequency.
    
    Parameters
    ----------
    df (pd.DataFrame):
        The dataframe that contains the data to be averaged.
    columns (list):
        The column names in the dataframe that need to be averaged.
    idx (int):
        The row to start averaging.
    model (str):
        The model to be used for averaging.
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