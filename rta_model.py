from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def main():
    exp_file = Path(r"C:\Users\Christine\Downloads\C0RAK595ML_Sample_11B.csv")
    sim_file = Path(r"C:\Users\Christine\Downloads\boron_total.csv")
    exp_df = pd.read_csv(exp_file)
    # exp_df =  
    sim_df = pd.read_csv(sim_file)

    exp_depth = pd.to_numeric(exp_df["Depth (nm).2"], errors='coerce')
    exp_conc = pd.to_numeric(exp_df["11B.1"], errors='coerce')
    # exp_conc = list(map(float, exp_df["11B"]))

    sim_depth = sim_df["depth (um)"] * 1E3
    temps = [col for col in sim_df.columns if col != "depth (um)"]

    best_temp = None
    min_error = float('inf')
    best_sim_aligned = None

    min_fit_depth = 10
    max_fit_depth = 90

    print(min(temps), max(temps))
    for temp in temps:
        sim_conc = sim_df[temp]

        f_interp = interp1d(sim_depth, sim_conc, kind='linear', bounds_error=False, fill_value=0)
        sim_conc_aligned = f_interp(exp_depth)

        log_exp = np.log10(exp_conc + 1e-10)
        log_sim = np.log10(sim_conc_aligned + 1e-10)
        
        # We only want to calculate error where we actually have experimental data
        valid_indices = (
            ~np.isnan(log_exp) & 
            ~np.isnan(log_sim) & 
            (exp_depth >= min_fit_depth) & 
            (exp_depth <= max_fit_depth)
        )
        
        if np.any(valid_indices):
            # error = np.mean(np.abs(exp_conc[valid_indices] - sim_conc_aligned[valid_indices]))
            error = np.mean((log_exp[valid_indices] - log_sim[valid_indices])**2)
            # error = np.trapezoid(np.log(exp_conc[valid_indices]/sim_conc_aligned[valid_indices]), np.unique(valid_indices))
            
            # Keep track of the lowest error
            if error < min_error:
                min_error = error
                best_temp = temp
                best_sim_aligned = sim_conc_aligned

    print(f"Estimated Annealing Temperature: {best_temp} (Log-MSE: {min_error:.4f})")
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(exp_depth, exp_conc, label="SIMS boron", color="C0", linewidth=2, marker=".")
    if best_temp is not None:
        ax.plot(exp_depth, best_sim_aligned, label=f"Best Fit Sim: {best_temp}", color="C1", linestyle="--")

    ax.legend()
    ax.set_yscale("log")
    ax.set_ylim(1E19, 6E20)
    ax.set_xlim(0, 100)
    ax.grid(True, which="both", alpha=0.5)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()