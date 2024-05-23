from os.path import splitext, basename

from PyLTSpice import SimRunner, SpiceEditor, LTspice, RawRead
from scipy.optimize import curve_fit, least_squares
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from postprocessing.parser import Parser


def electrical_ring_model(Rs: float, Rsi: float, Cp: float, Cj: float, Cox: float, netlist: SpiceEditor, filename: str):
    """
    The electrical equivalent model of a ring pn junction.

    Parameters
    ----------
    Rs : float
        Series resistance of the pn junction.
    Rsi : float
        Silicon substrate resistance in between the source and drain.
    Cp : float
        Pad capacitance between the electrodes.
    Cj : float
        Junction capacitance of the pn junction.
    Cox : float
        Oxide capacitance of the pn junction.
    """
    netlist.set_component_value('Rs', Rs)
    netlist.set_component_value('Rsi', Rsi)
    netlist.set_component_value('Cp', Cp)
    netlist.set_component_value('Cj', Cj)
    netlist.set_component_value('Cox', Cox)

    runner = SimRunner(simulator=LTspice)  # Configures the simulator to use and output
    runner.run_now(netlist, run_filename=f"{filename}.net")

    raw_data = RawRead(f"{filename}.raw")
    s11 = raw_data.get_trace("S11(v1)").get_wave()

    magnitude = 20*np.log10(np.abs(s11))
    phase = np.angle(s11)*180/np.pi

    return magnitude, phase

def residuals(params, netlist, filename, magnitude_obs, phase_obs):

    mag, phase = electrical_ring_model(*params, netlist, filename)

    distance = mag**2 + magnitude_obs**2 - 2*mag*magnitude_obs*np.cos(np.deg2rad(phase - phase_obs))

    return distance


def setup_netlist(spice_fp: str, src_name: str, freq: list):

    netlist = SpiceEditor(spice_fp)  # Open the Spice Model, and creates the .net
    netlist.set_element_model(src_name, "DC 0 AC 1")
    netlist.add_instructions(
        '; Simulation settings',
        f'.net {src_name} {src_name} rin=50',
        f'.ac list {" ".join(str(val) for val in freq)}',
    )

    return netlist

def search_space(*args: list, density: list):
    """
    Generate a search space for the parameters to be optimized.

    Parameters
    ----------
    *args : list
        A list of tuples containing the lower and upper bounds of the parameters.
    density : list
        The number of points to generate in each dimension of the search space.

    Yields
    ------
    tuple
        A tuple containing the parameters to be optimized.
    """
    arg_spaces = [np.linspace(*arg, dim) for arg, dim in zip(*args, density)]

    idx = -1
    while (idx:=idx+1) < np.prod(density):
        arg_idxs = np.unravel_index(idx, np.array(density))
        yield tuple(arg_space[arg_idx] for arg_space, arg_idx in zip(arg_spaces, arg_idxs))


def brute_force_search(netlist: SpiceEditor, spice_fp: str, s11: np.ndarray, s11_angle: np.ndarray):

    search_bounds = [(10, 100), (1e+03, 30e+03), (0.5e-15, 15e-15), (20e-15, 60e-15), (50e-15, 100e-15)]
    density = [6, 5, 5, 5, 5]
    results = []
    filename = splitext(spice_fp)[0]

    for params in tqdm(search_space(search_bounds, density=density), total=np.prod(density)):
        mag, phase = electrical_ring_model(*params, netlist, filename)
        distance = mag**2 + s11**2 - 2*mag*s11*np.cos(np.deg2rad(phase - s11_angle))
        results.append((*params, np.mean(np.sqrt(distance))))
        
    return pd.DataFrame(results, columns=["Rs", "Rsi", "Cp", "Cj", "Cox", "distance"])
    


def main():

    vs_name = "V1"
    spice_filepath = "postprocessing/ltspice/sparams.asc"
    filename = splitext(spice_filepath)[0]
    
    snp_filepath = "2024-4-25/ramzi_ring_b7_g200.s1p"

    # brute force searching for the initial guess
    # for snp_filepath in snp_filepaths:
    #     filename = splitext(basename(snp_filepath))[0]
    #     df = Parser.snp_parse(snp_filepath)
    #     freq = df["freq"]
    #     s11 = df["s11"]
    #     s11_angle = df["s11 angle"]

    #     netlist = setup_netlist(spice_filepath, vs_name, freq)

    #     df = brute_force_search(netlist, spice_filepath, s11, s11_angle)
    #     df.to_csv(f'brute_force_search_{filename.replace("-","_")}.csv', index=False)

    #     # find the row that has the mininum total error and print the parameters
    #     print(f'{filename}:\n{df.loc[df["distance"].idxmin()]}')

    # load the lowest initial guess 
    initial_guess = pd.read_csv("brute_force_search_ramzi_ring_b7_g200.csv")
    initial_guess = initial_guess.loc[initial_guess["distance"].idxmin(), ["Rs", "Rsi", "Cp", "Cj", "Cox"]].values

    df = Parser.snp_parse(snp_filepath)
    freq = df["freq"]
    s11 = df["s11"]
    s11_angle = df["s11 angle"]

    netlist = setup_netlist(spice_filepath, vs_name, freq)
    print(f"Initial guess: {initial_guess}")
    params = least_squares(residuals, initial_guess, args=(netlist, filename, s11, s11_angle)).x
    print("Optimized parameters: ", params)
    mag, phase = electrical_ring_model(*params, netlist, filename)
        
    freq = freq/1e+09
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    ax1.plot(freq, s11, color='r', label='Data')
    ax1.plot(freq, mag, 'r--', label='Fit')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Magnitude (dB)', color='r')
    ax1.set_ylim([-2.5, 0])
    ax1.legend()
    
    ax2.plot(freq, s11_angle, color='b', label='Data')
    ax2.plot(freq, phase, 'b--', label='Fit')
    ax2.set_ylabel('Phase (deg)', color='b')
    ax2.set_ylim([-70, 10])
    ax2.legend()
    fig.tight_layout()

    plt.show()

    # sparams = {
    #     "Rs": 37,
    #     "Rsi": 15e+03,
    #     "Cp": 1.55e-15,
    #     "Cj": 38e-15,
    #     "Cox": 78e-15,
    # }

    # mag, phase = electrical_ring_model(**sparams, netlist=netlist, filename=filename)
    # ax1.plot(freq, mag)
    # ax2.plot(freq, phase)
    # ax1.set_xlim([1, 67])


if __name__ == "__main__":
    main()
