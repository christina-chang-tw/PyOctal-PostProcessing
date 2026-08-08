import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from scipy.interpolate import griddata
from typing import List

@dataclass
class Data3D:
    xsize: int
    ysize: int
    pixel: float
    data: np.ndarray

@dataclass
class FitQuality:
    """Goodness-of-fit metrics between the measured data and the Zernike model."""
    mse:  float  # Mean Squared Error  (µm²)
    rmse: float  # Root Mean Squared Error (µm) — same units as height data
    r2:   float  # Coefficient of determination R² in [0, 1]
 
    def __str__(self) -> str:
        lines = [
            "--- Fit Quality ---",
            f"MSE:  {self.mse:.4f} µm²",
            f"RMSE: {self.rmse:.4f} µm",
            f"R²:   {self.r2:.6f}",
        ]
        return "\n".join(lines)

@dataclass
class ZernikeCoefficients:
    """
    Fitted Zernike coefficients (OSA/ANSI ordering).
    Terms not included in the fit are None.
    """
    # --- Order 0 (always fitted) ---
    piston:        float        # Z0  — constant height offset
 
    # --- Order 1 ---
    tilt_y:        float | None  # Z1  — hardware tilt along Y
    tilt_x:        float | None  # Z2  — hardware tilt along X
 
    # --- Order 2 ---
    astig_sin:     float | None  # Z3  — astigmatism / saddle (sin)
    defocus:       float | None  # Z4  — pure wafer bow / paraboloid
    astig_cos:     float | None  # Z5  — astigmatism / saddle (cos)
 
    # --- Order 3 ---
    coma_sin:      float | None  # Z6  — coma (sin) — off-centre stress
    coma_cos:      float | None  # Z7  — coma (cos) — off-centre stress
    trefoil_sin:   float | None  # Z8  — trefoil (sin) — 3-fold symmetry
    trefoil_cos:   float | None  # Z9  — trefoil (cos) — 3-fold symmetry
 
    # --- Order 4 ---
    astig2_sin:    float | None  # Z10 — 2nd astigmatism (sin)
    spherical:     float | None  # Z11 — primary spherical — edge roll-off
    astig2_cos:    float | None  # Z12 — 2nd astigmatism (cos)
    tetrafoil_sin: float | None  # Z13 — tetrafoil (sin) — 4-fold symmetry
    tetrafoil_cos: float | None  # Z14 — tetrafoil (cos) — 4-fold symmetry
 
    # --- Metadata ---
    n_terms:       int           = 1
    fit_quality:   FitQuality | None = None
 
    # ------------------------------------------------------------------
    # Severity helpers (only meaningful when both components are fitted)
    # ------------------------------------------------------------------
    @property
    def warp_severity(self) -> float | None:
        if self.astig_sin is None or self.astig_cos is None:
            return None
        return float(np.sqrt(self.astig_sin**2 + self.astig_cos**2))
 
    @property
    def coma_severity(self) -> float | None:
        if self.coma_sin is None or self.coma_cos is None:
            return None
        return float(np.sqrt(self.coma_sin**2 + self.coma_cos**2))
 
    # ------------------------------------------------------------------
    # Display — skips None terms automatically
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        def fmt(v: float | None) -> str:
            return f"{v:+.3f}" if v is not None else "—"
 
        lines = [f"--- Zernike Topography Breakdown (n={self.n_terms} terms) ---"]
 
        lines.append(f"  Piston (offset):      {fmt(self.piston)}")
 
        if self.tilt_y is not None or self.tilt_x is not None:
            lines.append(f"  Hardware tilt:        {fmt(self.tilt_y)}  {fmt(self.tilt_x)}")
 
        if self.defocus is not None:
            lines.append(f"  Defocus (bow):        {fmt(self.defocus)}")
 
        if self.astig_sin is not None or self.astig_cos is not None:
            sev = f"  (severity: {self.warp_severity:.3f})" if self.warp_severity else ""
            lines.append(f"  Astigmatism (warp):   {fmt(self.astig_sin)}  {fmt(self.astig_cos)}{sev}")
 
        if self.coma_sin is not None or self.coma_cos is not None:
            sev = f"  (severity: {self.coma_severity:.3f})" if self.coma_severity else ""
            lines.append(f"  Coma:                 {fmt(self.coma_sin)}  {fmt(self.coma_cos)}{sev}")
 
        if self.trefoil_sin is not None or self.trefoil_cos is not None:
            lines.append(f"  Trefoil:              {fmt(self.trefoil_sin)}  {fmt(self.trefoil_cos)}")
 
        if self.astig2_sin is not None or self.astig2_cos is not None:
            lines.append(f"  2nd Astigmatism:      {fmt(self.astig2_sin)}  {fmt(self.astig2_cos)}")
 
        if self.spherical is not None:
            lines.append(f"  Primary spherical:    {fmt(self.spherical)}")
 
        if self.tetrafoil_sin is not None or self.tetrafoil_cos is not None:
            lines.append(f"  Tetrafoil:            {fmt(self.tetrafoil_sin)}  {fmt(self.tetrafoil_cos)}")
 
        if self.fit_quality is not None:
            lines += ["", str(self.fit_quality)]
 
        return "\n".join(lines)
    

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
    z_matrix = raw_array.reshape((data.xsize, data.ysize))
    z_matrix[z_matrix > 1e20] = np.nan
    data.data = z_matrix

    return data
 

 
 
# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------
 
def _build_design_matrix(rho: np.ndarray, theta: np.ndarray, n_terms: int) -> np.ndarray:
    """
    Build the Zernike design matrix for the first n_terms standard terms.
 
    Parameters
    ----------
    rho     : Normalised radius in [0, 1].
    theta   : Azimuthal angle in radians.
    n_terms : Number of terms to include — must be one of VALID_N_TERMS.
 
    Returns
    -------
    A : ndarray, shape (N, n_terms)
    """
    r2 = rho**2
    r3 = rho**3
    r4 = rho**4
 
    all_terms = [
        # Order 0
        np.ones_like(rho),                                        # Z0  piston
        # Order 1
        2 * rho * np.sin(theta),                                  # Z1  tilt Y
        2 * rho * np.cos(theta),                                  # Z2  tilt X
        # Order 2
        np.sqrt(6)  * r2 * np.sin(2*theta),                      # Z3  astig sin
        np.sqrt(3)  * (2*r2 - 1),                                 # Z4  defocus
        np.sqrt(6)  * r2 * np.cos(2*theta),                      # Z5  astig cos
        # Order 3
        np.sqrt(8)  * (3*r3 - 2*rho) * np.sin(theta),            # Z6  coma sin
        np.sqrt(8)  * (3*r3 - 2*rho) * np.cos(theta),            # Z7  coma cos
        np.sqrt(8)  * r3 * np.sin(3*theta),                      # Z8  trefoil sin
        np.sqrt(8)  * r3 * np.cos(3*theta),                      # Z9  trefoil cos
        # Order 4
        np.sqrt(10) * (4*r4 - 3*r2) * np.sin(2*theta),           # Z10 2nd astig sin
        np.sqrt(5)  * (6*r4 - 6*r2 + 1),                         # Z11 spherical
        np.sqrt(10) * (4*r4 - 3*r2) * np.cos(2*theta),           # Z12 2nd astig cos
        np.sqrt(10) * r4 * np.sin(4*theta),                       # Z13 tetrafoil sin
        np.sqrt(10) * r4 * np.cos(4*theta),                       # Z14 tetrafoil cos
    ]
 
    return np.column_stack(all_terms[:n_terms])
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def fit_zernike(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_terms: int = 6,
    max_radius: float | None = None,
) -> ZernikeCoefficients:
    """
    Fit Zernike coefficients to scattered (x, y, z) surface data.
 
    Parameters
    ----------
    x, y       : Cartesian coordinates of the measurement points (mm).
    z          : Height values at each point (µm).
    n_terms    : Number of Zernike terms to fit. Must be one of: 1, 3, 6, 10, 15.
                 Terms are always added as complete radial orders:
                   1  → order 0 only
                   3  → orders 0-1
                   6  → orders 0-2  (default)
                   10 → orders 0-3
                   15 → orders 0-4
    max_radius : Radius used to normalise rho to [0, 1].
                 Defaults to the maximum radial distance in (x, y).
 
    Returns
    -------
    ZernikeCoefficients  (unfitted terms are None; includes FitQuality)
    """
    # if n_terms not in VALID_N_TERMS:
    #     raise ValueError(f"n_terms must be one of {VALID_N_TERMS}, got {n_terms}.")
 
    r = np.sqrt(x**2 + y**2)
    if max_radius is None:
        max_radius = float(np.max(r))
 
    rho   = r / max_radius
    theta = np.arctan2(y, x)
 
    A = _build_design_matrix(rho, theta, n_terms)
    C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
 
    # Goodness-of-fit metrics
    residuals = z - A @ C
    mse    = float(np.mean(residuals**2))
    rmse   = float(np.sqrt(mse))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((z - np.mean(z))**2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
 
    # Helper: return C[i] if that term was fitted, else None
    def get(i: int) -> float | None:
        return float(C[i]) if i < n_terms else None
 
    return ZernikeCoefficients(
        piston        = float(C[0]),
        tilt_y        = get(1),
        tilt_x        = get(2),
        astig_sin     = get(3),
        defocus       = get(4),
        astig_cos     = get(5),
        coma_sin      = get(6),
        coma_cos      = get(7),
        trefoil_sin   = get(8),
        trefoil_cos   = get(9),
        astig2_sin    = get(10),
        spherical     = get(11),
        astig2_cos    = get(12),
        tetrafoil_sin = get(13),
        tetrafoil_cos = get(14),
        n_terms       = n_terms,
        fit_quality   = FitQuality(mse=mse, rmse=rmse, r2=r2),
    )
 
 
def reconstruct_surface(
    coeffs: ZernikeCoefficients,
    max_radius: float,
    grid_points: int = 1000,
    remove_piston_tilt: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the fitted Zernike surface on a regular Cartesian grid.
 
    Only terms that were actually fitted (non-None) are included.
 
    Parameters
    ----------
    coeffs             : Fitted ZernikeCoefficients.
    max_radius         : Physical radius of the wafer (mm) — sets the grid extent.
    grid_points        : Number of grid points along each axis.
    remove_piston_tilt : When True (default), omit Z0/Z1/Z2 so the surface
                         is levelled before plotting.
 
    Returns
    -------
    grid_x, grid_y : 2-D coordinate grids (mm).
    grid_z         : Reconstructed height map (µm); NaN outside the wafer.
    """
    linear = np.linspace(-max_radius, max_radius, grid_points)
    grid_x, grid_y = np.meshgrid(linear, linear, indexing="ij")
 
    grid_r     = np.sqrt(grid_x**2 + grid_y**2)
    grid_rho   = grid_r / max_radius
    grid_theta = np.arctan2(grid_y, grid_x)
 
    r2 = grid_rho**2
    r3 = grid_rho**3
    r4 = grid_rho**4
 
    grid_z = np.zeros_like(grid_x)
 
    # Helper: add a term only if its coefficient was fitted
    def add(coeff: float | None, basis: np.ndarray) -> None:
        if coeff is not None:
            grid_z.__iadd__(coeff * basis)
 
    if not remove_piston_tilt:
        add(coeffs.piston,   np.ones_like(grid_rho))
        add(coeffs.tilt_y,   2 * grid_rho * np.sin(grid_theta))
        add(coeffs.tilt_x,   2 * grid_rho * np.cos(grid_theta))
 
    add(coeffs.astig_sin,     np.sqrt(6)  * r2 * np.sin(2*grid_theta))
    add(coeffs.defocus,       np.sqrt(3)  * (2*r2 - 1))
    add(coeffs.astig_cos,     np.sqrt(6)  * r2 * np.cos(2*grid_theta))
    add(coeffs.coma_sin,      np.sqrt(8)  * (3*r3 - 2*grid_rho) * np.sin(grid_theta))
    add(coeffs.coma_cos,      np.sqrt(8)  * (3*r3 - 2*grid_rho) * np.cos(grid_theta))
    add(coeffs.trefoil_sin,   np.sqrt(8)  * r3 * np.sin(3*grid_theta))
    add(coeffs.trefoil_cos,   np.sqrt(8)  * r3 * np.cos(3*grid_theta))
    add(coeffs.astig2_sin,    np.sqrt(10) * (4*r4 - 3*r2) * np.sin(2*grid_theta))
    add(coeffs.spherical,     np.sqrt(5)  * (6*r4 - 6*r2 + 1))
    add(coeffs.astig2_cos,    np.sqrt(10) * (4*r4 - 3*r2) * np.cos(2*grid_theta))
    add(coeffs.tetrafoil_sin, np.sqrt(10) * r4 * np.sin(4*grid_theta))
    add(coeffs.tetrafoil_cos, np.sqrt(10) * r4 * np.cos(4*grid_theta))
 
    grid_z[grid_r > max_radius] = np.nan
 
    return grid_x, grid_y, grid_z

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

    data = read_wyko_asc(input_fp)
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