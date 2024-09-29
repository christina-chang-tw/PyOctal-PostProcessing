import numpy as np

def a_to_db_alpha(a: float, length: float) -> float:
    """
    Convert the amplitude coefficient to the loss coefficient in dB/cm.
    
    Args:
        a (float): The amplitude coefficient.
        length (float): The length of the ring resonator.
    
    Returns:
        float: The loss coefficient in dB/cm.
    """
    alpha_linear = -2/length*np.log(a)
    return 10*np.log10(alpha_linear)

def db2w(val: float) -> float:
    """
    Convert a db value to a linear value.
    
    Args:
        val (float): The value in dB/dbm.
    """
    return np.power(10, val/10)