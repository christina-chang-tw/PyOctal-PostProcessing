import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from postprocessing.parser import Parser, Data3D
from postprocessing.zernike import fit_zernike, reconstruct_surface, ZernikeCoefficients


def plot_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 radius: float, ysize: float, angles: np.ndarray, min_max: List, filename: Path):
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    heatmap = ax.pcolormesh(x, y, z, cmap='coolwarm', shading='auto', vmin=min_max[0], vmax=min_max[1])
    # heatmap = ax.pcolormesh(x, y, z, cmap='coolwarm', shading='auto')

    for i in range(ysize):
        x_spoke = radius * np.cos(angles[i])
        y_spoke = radius * np.sin(angles[i])
        ax.plot(x_spoke, y_spoke, color='black', alpha=0.3, linewidth=1, linestyle='--')

    fig.colorbar(heatmap, ax=ax, shrink=0.7, label='Height (um)')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(filename, dpi=400)

    h_min = np.nanmin(z)
    h_max = np.nanmax(z)
    print(f"H min (um): {round(h_min, 2)}, H max (um): {round(h_max, 2)}, H diff (um): {round(h_max - h_min, 2)}")


def analysis(data: Data3D, angles: np.ndarray, n_terms: int=6):
    x, y, z = [], [], []
    radius = np.arange(data.xsize) * data.pixel # in mm

    for i in range(data.ysize):
        z_radial = data.data[:,i] - data.data[0,i] # assume that the centre meets.
        x_radial = radius * np.cos(angles[i])
        y_radial = radius * np.sin(angles[i])

        x.extend(x_radial)
        y.extend(y_radial)
        z.extend(z_radial)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    max_radius = np.max(np.sqrt(x**2 + y**2))
    coeffs = fit_zernike(x, y, z, n_terms=n_terms, max_radius=max_radius)

    print(f"Coeffs: {coeffs}")

    grid_x, grid_y, grid_z = reconstruct_surface(coeffs, max_radius)
    return grid_x, grid_y, grid_z


def main():
    input_fp = Path(r"C:\Users\Christine\Downloads\RCA_setupv2_wafer_post1.ASC")
    output_fp = Path(r"./bowing.png")
    rotation_degree = 30
    rotation_offset = 0

    data = Parser.wyko_asc_parse(input_fp)
    degs = np.arange(0, data.ysize * rotation_degree, rotation_degree) - rotation_offset
    angles = np.radians(degs)
    radius = np.arange(data.xsize) * data.pixel

    grid_x, grid_y, grid_z = analysis(data, angles, n_terms=10)
    plot_surface(grid_x, grid_y, grid_z, radius, data.ysize,
                 angles=angles, min_max=[None, None], filename=output_fp)
    # coeffs = ZernikeCoefficients(0, 0, 0, 150, 0, -150)
    # x, y, z = reconstruct_surface(coeffs, 1)
    # plot_surface(x, y, z, 1, angles.size, angles)

    plt.show()

if __name__ == "__main__":
    main()
