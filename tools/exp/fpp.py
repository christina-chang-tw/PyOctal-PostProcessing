"""
Wafer Sheet Resistance Contour Plot
====================================
Parses a Four Dimensions 200mm wafer map (121-point layout) and produces
a colour contour plot of sheet resistance (Ohm/sq) vs. x, y position.

The 121-point map uses a standard concentric-ring layout:
  Circle 0 (centre): 1 point
  Circle 1          : 8 points
  Circle 2          : 16 points
  Circle 3          : 24 points   (file labels circles differently – see below)
  Circle 4          : 32 points
  Circle 5          : 40 points

The file groups are labelled Circle 1-5, but the very first point (130.575)
belongs to the single centre point.  Radii are scaled to keep the outermost
ring within the 180 mm maximum test diameter (90 mm radius).
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from pathlib import Path

# ---------------------------------------------------------------------------
# 1.  Read the raw data file
# ---------------------------------------------------------------------------
DATA_FILE = Path(r"C:\Users\Christine\Downloads\RTA_dev2_slot25.txt")          # change path if needed

raw_text = DATA_FILE.read_text()

# ---------------------------------------------------------------------------
# 2.  Extract header statistics for annotation
# ---------------------------------------------------------------------------
def _hdr(key, text):
    m = re.search(rf"{re.escape(key)}\s*:\s*([\S ]+)", text)
    return m.group(1).strip() if m else "N/A"

wafer_id   = _hdr("Wafer ID",       raw_text)
date_str   = _hdr("Date (M/D/Y)",   raw_text)
mean_val   = _hdr("Mean",           raw_text)
std_val    = _hdr("Std Deviation",  raw_text)
min_max    = _hdr("Min, Max",       raw_text)
uniformity = _hdr("Uniformity",     raw_text)
contour    = _hdr("Contour",        raw_text)

# ---------------------------------------------------------------------------
# 3.  Extract measurement values (skip deleted points marked with *)
# ---------------------------------------------------------------------------
#  Lines look like:  "84.   187.3125*" or "86.   148.5000"
pattern = re.compile(r"(\d+)\.\s+([\d.]+)(\*?)")
points_raw = pattern.findall(raw_text)           # [(idx, value, deleted_flag), ...]

indices = []
values  = []
deleted = []

for idx_s, val_s, flag in points_raw:
    if float(val_s) > 132:
        flag = "*"

    indices.append(int(idx_s))
    values.append(float(val_s))
    deleted.append(flag == "*")

n_total = len(values)   # should be 121

# ---------------------------------------------------------------------------
# 4.  Build (x, y) coordinates for the 121-point map
#
#     Standard Four Dimensions 200 mm / 121-pt layout:
#       Ring 0 (centre, 1 pt)  : r = 0 mm
#       Ring 1 (8 pts)         : r ≈ 20 mm
#       Ring 2 (16 pts)        : r ≈ 40 mm
#       Ring 3 (24 pts)        : r ≈ 60 mm
#       Ring 4 (32 pts)        : r ≈ 80 mm
#       Ring 5 (40 pts)        : r ≈ 90 mm  (within 90 mm test radius)
#
#     First point in each ring starts at 90° (top), proceeding clockwise
#     (matching the file's Circle labelling convention).
# ---------------------------------------------------------------------------

ring_defs = [
    # (n_points, radius_mm, start_angle_deg, direction)
    (1,  0.0,  90.0, 1),   # centre
    (8,  20.0, 90.0, 1),   # ring 1  – file "Circle 1" contains pt 1 (centre) + ring1
    (16, 40.0, 90.0, 1),   # ring 2
    (24, 60.0, 90.0, 1),   # ring 3
    (32, 80.0, 90.0, 1),   # ring 4
    (40, 90.0, 90.0, 1),   # ring 5
]

coords_x = []
coords_y = []

for n_pts, radius, start_deg, direction in ring_defs:
    if n_pts == 1:
        coords_x.append(0.0)
        coords_y.append(0.0)
    else:
        step = 360.0 / n_pts
        for k in range(n_pts):
            angle_deg = start_deg - direction * k * step   # clockwise
            angle_rad = np.radians(angle_deg)



            coords_x.append(radius * np.cos(angle_rad))
            coords_y.append(radius * np.sin(angle_rad))

coords_x = np.array(coords_x)
coords_y = np.array(coords_y)
values   = np.array(values)
deleted  = np.array(deleted)

# Separate good and deleted points
mask_good = ~deleted
x_good    = coords_x[mask_good]
y_good    = coords_y[mask_good]
v_good    = values[mask_good]

x_del     = coords_x[deleted]
y_del     = coords_y[deleted]
v_del     = values[deleted]

# ---------------------------------------------------------------------------
# 5.  Interpolate onto a fine grid using Delaunay triangulation
# ---------------------------------------------------------------------------
triang = mtri.Triangulation(x_good, y_good)

# Mask triangles that fall outside the wafer test radius (90 mm)
cx = np.mean(triang.x[triang.triangles], axis=1)
cy = np.mean(triang.y[triang.triangles], axis=1)
triang.set_mask(np.sqrt(cx**2 + cy**2) > 91)

xi = np.linspace(-92, 92, 400)
yi = np.linspace(-92, 92, 400)
Xi, Yi = np.meshgrid(xi, yi)

interp = mtri.LinearTriInterpolator(triang, v_good)
Zi = interp(Xi, Yi)

# Mask grid points outside wafer edge
wafer_radius = 90.0
outside = np.sqrt(Xi**2 + Yi**2) > wafer_radius
Zi[outside] = np.nan

# ---------------------------------------------------------------------------
# 6.  Colour map  –  blue (low) → green → yellow → red (high)
# ---------------------------------------------------------------------------
# cmap_colors = ["#1a4fa0", "#3a8ad4", "#50c8aa", "#a8d96c",
#                "#f5e642", "#f5a623", "#e8402a", "#8b0000"]
# custom_cmap = LinearSegmentedColormap.from_list("wafer_rs", cmap_colors, N=256)

# ---------------------------------------------------------------------------
# 7.  Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_facecolor("#0f1923")

vmin = v_good.min()
vmax = v_good.max()

cf = ax.contourf(Xi, Yi, Zi, levels=30,
                 vmin=vmin, vmax=vmax, extend="both")
cs = ax.contour(Xi, Yi, Zi, levels=12, colors="black",
                linewidths=0.4, alpha=0.35)
ax.clabel(cs, fmt="%.1f", fontsize=10, colors="black", inline=True)

# Wafer edge
wafer_circle = Circle((0, 0), wafer_radius, fill=False,
                       edgecolor="black", linewidth=1.5, linestyle="--", zorder=4)
ax.add_patch(wafer_circle)

# Flat/notch indicator at bottom
notch_x = np.linspace(-15, 15, 100)
notch_y = np.full_like(notch_x, -(wafer_radius))
ax.plot(notch_x, notch_y, color="black", linewidth=3, zorder=5)

# Scatter – good measurement points
sc = ax.scatter(x_good, y_good, c=v_good,
                vmin=vmin, vmax=vmax,
                s=18, edgecolors="black", linewidths=0.5, zorder=5)

# Scatter – deleted points (marked with *)
if len(x_del) > 0:
    ax.scatter(x_del, y_del, c="red",
               s=80, linewidths=1.8,
               marker="x", zorder=6, label="Deleted (*)")
    ax.legend(loc="upper right", fontsize=8,
              edgecolor="black", labelcolor="black")

# Colour bar
cbar = fig.colorbar(cf, ax=ax, fraction=0.038, pad=0.03, extend="both")
cbar.set_label("Sheet Resistance  (Ω/sq)", color="black", fontsize=10, labelpad=10)
cbar.ax.yaxis.set_tick_params(color="black")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black", fontsize=8)
cbar.outline.set_edgecolor("black")

# Axis labels & ticks
ax.set_xlabel("X Position  (mm)", color="black", fontsize=10, labelpad=8)
ax.set_ylabel("Y Position  (mm)", color="black", fontsize=10, labelpad=8)
ax.tick_params(colors="black", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#445566")

ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_aspect("equal")
ax.grid(True, color="#223344", linewidth=0.5, alpha=0.5)

# # Title block
# fig.text(0.13, 0.955, wafer_id, color="black",
#          fontsize=13, fontweight="bold", va="top", fontfamily="monospace")
# fig.text(0.13, 0.932, f"Date: {date_str}    Layer: P    Thickness: 0.1 µm",
#          color="#aabbcc", fontsize=8, va="top", fontfamily="monospace")

stats_text = (
    f"Mean:        {mean_val}\n"
    f"Std Dev:     {std_val}\n"
    f"Min / Max:   {min_max}\n"
    f"Uniformity:  {uniformity}\n"
    f"Contour:     {contour}"
)
fig.text(0.78, 0.955, stats_text, color="#aaddff",
         fontsize=7.5, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#152030",
                   edgecolor="#335566", alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = "wafer_sheet_resistance_contour.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Plot saved → {out_path}")
plt.show()