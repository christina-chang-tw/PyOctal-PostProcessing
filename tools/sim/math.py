import numpy as np

def ring_phase_based_on_influence_Xiuyou_Han(a, t, phi):

    if a == t: # critical coupling
        phase_response = np.arctan2(np.sin(phi),(1-np.cos(phi))) - np.arctan2(a**2*np.sin(phi),(1-a**2*np.cos(phi)))
    elif a > t: # overcoupling
        phase_response = np.pi - phi - np.arctan2(t*np.sin(phi),(a-t*np.cos(phi))) - np.arctan2(t*a*np.sin(phi),(1-t*a*np.cos(phi)))
    else: # undercoupling
        phase_response = np.arctan2(a*np.sin(phi),(t-a*np.cos(phi))) - np.arctan2(t*a*np.sin(phi),(1-t*a*np.cos(phi)))

    return phase_response

def ring_intensity(a, t, phi):
    return (a**2 + t**2 - 2*a*t*np.cos(phi))/(1 + a**2*t**2 - 2*a*t*np.cos(phi))

def ring_phase_based_on_microresonator_wim(a, t, phi):
    phase_response = np.pi + phi + np.arctan2((t*a*np.sin(phi)),(1-t*a*np.cos(phi))) + np.arctan2((t*np.sin(phi)),(a-t*np.cos(phi)))
    return phase_response

def ring_phase_based_on_microresonator_mine(a, t, phi):
    phase_response = np.arctan2(a*(1-t**2)*np.sin(phi), t*(1+a**2)-a*(1+t**2)*np.cos(phi))
    return phase_response

def ramzi_output_intensity(dphi, a, t, phi):

    ring_ph = np.pi + phi + np.arctan2((t*a*np.sin(phi)),(1-t*a*np.cos(phi))) + np.arctan2((t*np.sin(phi)),(a-t*np.cos(phi)))

    field = (t-a*np.exp(1j*phi))/(1-t*a*np.exp(1j*phi))
    transmission = (a**2 - 2*a*t*np.cos(phi) + t**2) / (1 - 2*a*t*np.cos(phi) + (a*t)**2)

    return (transmission + 1 + 2*np.abs(field)*np.cos(dphi + ring_ph))/4

def mzi_output(
    I0: float,
    delta_phi: np.array,
) -> float:
    return I0/2*(1 + np.cos(delta_phi))

def mzi_delta_phi(
    neff: float,
    wavelength: float,
) -> float:
    return 2*np.pi*neff/wavelength