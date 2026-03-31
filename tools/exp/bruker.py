import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter1d

@dataclass
class Data3D:
    xsize: int
    ysize: int
    pixel: float
    data: np.ndarray


def read_wyko_asc(filepath: Path):
    data = Data3D(0,0,0,[0])
    data_start_line = 0

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse the header
    for i, line in enumerate(lines):
        parts = line.split() 
        if line.startswith("X Size"):
            data.xsize = int(parts[-1])
        elif line.startswith("Y Size"):
            data.ysize = int(parts[-1])
        elif line.startswith("Pixel_size"):
            data.pixel = float(parts[-1])
        elif line.startswith("RAW_DATA"):
            data_start_line = i + 1
            break

    raw_array = np.loadtxt(lines[data_start_line:]) / 1E3 # nm to um
    z_matrix = raw_array.reshape((data.ysize, data.xsize))
    z_matrix[z_matrix > 1e20] = np.nan

    # filtering the noise
    window_size = round(data.xsize * 0.05)
    data.data = uniform_filter1d(z_matrix, size=window_size, axis=1)

    return data


def analysis_3d(data: Data3D, angles: np.ndarray):
    x, y, z = [], [], []
    radius = np.arange(data.xsize) * data.pixel # in mm

    for i in range(data.ysize):
        z_radial = data.data[i, :] - data.data[i, 0]
        x_radial = radius * np.cos(angles[i])
        y_radial = radius * np.sin(angles[i])

        x.extend(x_radial)
        y.extend(y_radial)
        z.extend(z_radial)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    valid_mask = ~np.isnan(z)
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

    grid_x, grid_y = np.mgrid[-max(radius):max(radius):5000j, 
                              -max(radius):max(radius):5000j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    distance_from_center = np.sqrt(grid_x**2 + grid_y**2)
    grid_z[distance_from_center > max(radius)] = np.nan

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(data.ysize):
        x_spoke = radius * np.cos(angles[i])
        y_spoke = radius * np.sin(angles[i])
        z_spoke = data.data[i, :] - data.data[i, 0] # Aligned Z
        ax.plot(x_spoke, y_spoke, z_spoke, color='black', alpha=0.3, linewidth=1)

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='coolwarm', edgecolor='none', alpha=0.7)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Height (um)')
    ax.set_title("Interpolated Wafer Bow")
    ax.set_xlabel("X (mm)")
    ax.set_xlim(-max(radius), max(radius))
    ax.set_ylabel("Y (mm)")
    ax.set_ylim(-max(radius), max(radius))
    ax.set_zlabel("Height (um)")
    ax.set_box_aspect((1, 1, 0.4))
    fig.tight_layout()


def analysis_2d(data: Data3D, angles: np.ndarray):
    radius = np.arange(data.xsize) * data.pixel
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for i in range(data.ysize):
        z_spoke = data.data[i, :] - data.data[i, 0] # Aligned Z
        ax.plot(radius, z_spoke, label=f"{np.round(angles[i]*180/np.pi)} deg")

    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Height (um)")
    ax.set_xlim(0, max(radius))
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    fig.tight_layout()


def main():
    fp = Path(r"E:\3d_data.ASC")
    rotation_degree = 30

    data = read_wyko_asc(fp)
    angles = np.radians(np.arange(0, data.ysize * 30, rotation_degree))
    analysis_2d(data, angles)
    analysis_3d(data, angles)

    plt.show()

if __name__ == "__main__":
    main()