"""
vcsv.py

Deal with vcsv files from Virtuoso Cadence.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class VCSV:
    def __init__(self, filename: str):
        with open(filename, "r", encoding="utf-8") as file:
            self.lines = file.readlines()

    def parse(self):
        """
        Parse the VCSV file.
        """
        data = dict()
        labels = self.lines[1].split(",;")
        df = pd.DataFrame([data.split(",") for data in self.lines[6:]], dtype=float)
        for idx, label in enumerate(labels):
            data[label] = np.round((df[idx*2], df[idx*2+1]), 9)

        return data


def main():
    filename = "nmos-wr-400n-l-30n-nr-4-m-1-ids-vgs-sweep.vcsv"
    vc = VCSV(filename)
    data = vc.parse()

    for key, val in data.items():
        plt.plot(val[0], val[1])

    plt.show()

if __name__ == "__main__":
    main()