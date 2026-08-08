from pathlib import Path
from typing import List
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# define your shot map here
SHOT_MAP = [
    [0,0,0,0,1,1,2,1,1,2,1,0,0,0,0],
    [0,0,2,1,1,2,1,1,2,1,1,2,1,0,0],
    [0,2,1,1,2,1,1,2,1,1,2,1,1,2,0],
    [2,1,1,2,1,1,2,1,1,2,1,1,2,1,1],
    [1,1,2,1,1,2,1,1,2,1,1,2,1,1,2],
    [1,2,1,1,2,1,1,2,1,1,2,1,1,2,1],
    [2,1,1,2,1,1,2,1,1,2,1,1,2,1,1],
    [1,1,2,1,1,2,1,1,2,1,1,2,1,1,2],
    [0,2,1,1,2,1,1,2,1,1,2,1,1,2,0],
    [0,0,1,2,1,1,2,1,1,2,1,1,2,0,0],
    [0,0,0,0,1,2,1,1,2,1,1,0,0,0,0]
]

def rotate_180_degrees(x, y):
    return -x, -y

def main():
    # Compute everything in um!!!
    # The plot assumes notch facing towards the machine!!!
    chip_spacing = (12500, 16500) # (x, y)
    recenter_col = (-5, -7) # need to reverse the SHOT_MAP because it shoudl now be from bottom up for recentering the row
    SHOT_MAP.reverse()

    points_of_interest = { # a single point of interest on each shot map delta to center
        1: (895, -558.4),
        2: (728.1, -3322.3)
    }

    stored_points = []

    fig, ax = plt.subplots(1, 1, figsize=(25, 10))

    for i, row in enumerate(SHOT_MAP): # i is the row
        i = i + recenter_col[0]
        for j, quad in enumerate(row):
            if quad == 0:
                continue
            elif quad in points_of_interest.keys():
                j = j + recenter_col[1]

                # print(i, j, chip_spacing[0] * j, chip_spacing[1] * i)
                x = points_of_interest[quad][0] + chip_spacing[0] * j
                y = points_of_interest[quad][1] + chip_spacing[1] * i

                x, y = rotate_180_degrees(x, y)
                stored_points.append([i, j, quad, round(x, 1), round(y, 1)])
                ax.scatter(j, i, c="black", s=10, alpha=0.5)
                ax.annotate(
                    f"({x:.0f},{y:.0f})", 
                    (j, i), 
                    fontsize=6, 
                    ha='center', 
                    va='bottom',
                    xytext=(0, 0), 
                    textcoords='offset points'
                )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_ylim(6, -6)
    ax.set_xlim(8, -8)
    # fig.tight_layout()
    fig.savefig("plot.png", dpi=400)


    df = pd.DataFrame(stored_points, columns=["Row", "Column", "Quadrant", "X (um)", "Y (um)"])
    df.to_csv("stylus_map.csv", index=False)
    plt.show()

if __name__ == "__main__":
    main()                

