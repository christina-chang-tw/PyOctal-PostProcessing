import numpy as np

def a_to_db_alpha(a, length):
    """Converts the amplitude coefficient to the loss coefficient in dB/cm.
    
    Args:
        a (float): The amplitude coefficient.
        length (float): The length of the ring resonator.
    
    Returns:
        float: The loss coefficient in dB/cm.
    """
    alpha_linear = -2/length*np.log(a)
    return 10*np.log10(alpha_linear)

def db2w(val: float):
    return np.power(10, val/10)