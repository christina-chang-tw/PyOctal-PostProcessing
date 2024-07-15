from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from postprocessing.parser import Parser
from postprocessing.utils.formatter import Publication

Publication.set_basics()

fig, ax = plt.subplots(1, 1)

file = r"C:\Users\cchan\Downloads\amplifier_bandwidth_amp.vcsv"

a = Parser.vcsv_parse(Path(file))

for _, value in a.items():
    ax.plot(value[0]/1E+09, value[1])

ax.set_ylim([0, 26])
ax.set_xlim([0.08, 60])
ax.set_xscale("log")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Gain [dB]")
ax.legend(["RF+L", "RF", "Ideal"])
fig.savefig("amp.pdf")
plt.show()
