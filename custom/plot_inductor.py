import numpy as np
import matplotlib.pyplot as plt

from postprocessing.parser import Parser
from postprocessing.utils.formatter import Publication


def main():
    Publication.set_basics()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # I want to plot from a pdf image of the inductor - it is the inductor structure
    ax[0].imshow(plt.imread(r"C:\Users\cchan\Pictures\Picture4.png"))
    # disable axis
    ax[0].axis("off")


    length = [30, 40, 50, 60, 70]
    for l in length:
        file = r"C:\Users\cchan\Downloads\TYC_IND_W2um_L{}um_GAP1_M9.csv".format(l)
        data = Parser.ads_parse(file)
        ax[1].plot(data["freq"]/1E+09, data["Leff"]*1E+12, label=f"L={l}um")
        ax[1].set_xlabel("Frequency [GHz]")
        ax[1].set_ylabel("Inductance [pH]")
    ax[1].set_ylim(0, 600)
    ax[1].set_xlim(0.1, 60)
    ax[1].legend()
    ax[0].set_title("(a)")
    ax[1].set_title("(b)")
    fig.tight_layout()
    fig.savefig("inductor.pdf")
    plt.show()

if __name__ == '__main__':
    main()