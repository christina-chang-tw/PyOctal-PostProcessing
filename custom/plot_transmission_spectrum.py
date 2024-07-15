import pandas as pd
import matplotlib.pyplot as plt

voltages = [0, 1, 2, 3, 4]

for volt in voltages:
    # 9 and 15
    file = f"2024-2-27-ring/Radius4_t19_{volt}v.csv"
    df = pd.read_csv(file)
    plt.plot(df["Wavelength"]*1E+09, -df["Loss [dB]"], label=f"{volt}V")

plt.xlim(1545, 1550)
# plt.ylim(-35, -25)
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Loss [dB]")
plt.show()
