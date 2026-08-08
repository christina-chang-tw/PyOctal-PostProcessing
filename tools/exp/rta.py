from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def historical():
    # folder = Path(r"C:\Users\Christine\Downloads")
    folder = Path(r"D:\RTA\Christina")    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    # ax2 = ax.twinx()
    # lists = [1, 3, 5, 6]
    # lists = [7, 8, 9, 10, 11, 12, 13, 14, 16, 18]
    # lists = [15, 17, 19, 21]
    # lists = ["200mm_SOI_1000C_setupv2_slot9.HIS", "200mm_SOI_1050C_setupv2_slot11.HIS", "200mm_SOI_1100C_setupv2_slot13.HIS", "200mm_SOI_1150C_setupv2_slot1.HIS"]
    # lists = ["200mm_SI_Oxi_test_slot1(30sec).HIS"]
    lists = [r"200mm_SOI_Yifu_1000C_20260728144.HIS"]
    # lists = [r"E:\RTA\Georgia\200mm_SI_Oxi_test_slot7_(120sec).HIS", r"E:\RTA\Christina\200mm_RTA_dev3_slot3_1050C.HIS"]
    labels = ["5nm SiN"]

    # df0 = pd.read_csv(lists[0], skiprows=9)
    ax2 = ax.twinx()


    for l, label, c in zip(lists, labels, ["C0", "C1"]):
        file = Path(folder / l)
        df = pd.read_csv(file, skiprows=9)
        line1 = ax.plot(df["Time"], df["Pressure"], label=f"Pressure - {label}", linestyle='-', color="C0")
        # ax.plot(df["Time"], df["O2"], label=f"O2 - {label}", linestyle='--', color=c)
        # ax.plot(df["Time"], df["H2"], label=f"H2 - {label}", linestyle='--', color=c)
        # line3 = ax.plot(df["Time"], df["N2"], label=f"N2", linestyle='-', color=c)
        line2 = ax2.plot(df["Time"], df["Power"], label=f"Power - {label}", linestyle='-', color="C1")
        
        line4 = ax.plot(df["Time"], df["Pyro"], label=f"Temp. - {label}", linestyle='-', color="C2")
        ax.plot(df["Time"], df["Setpoint"], label=f"Set Temp. - {label}", linestyle='-.', color="C3")
        # ax.plot(df["Time"], df["BVSetpoint"], label=f"BV Setpoint - {label}", linestyle='-', marker='o', color=c)
        # ax.plot(df["Time"], df["BVPressure"], label=f"BV Pressure - {label}", linestyle='-', marker='o', color=c)
        # ax.plot(df["Time"], df["HighVacuum"], label=f"High vacuum - {label}", linestyle=':', color=c)
        # ax.plot(df["Time"], df["BVAngle"], label=f"BV Angle - {label}", linestyle='-', marker='o', color=c)
    
    # ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.5)
    ax.legend()
    # ax2.legend(bbox_to_anchor=(1.1, 0.1), loc='upper left', borderaxespad=0.5)
    ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Temperature (C)")
    ax.set_ylabel("Temperature (C)/Pressure (mbar)")
    ax2.set_ylabel("Power %")
    # ax.set_ylabel("Temperature (C)/Pressure (mbar)")
    ax.set_xlim(0, 600)
    ax2.set_ylim(0, 100)
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig("plot.png", dpi=400)
    # plt.show()


def recipe():
    folder = Path(r"E:\RTA\recipes")
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    lists = ["200mm_SOI_Yifu_1000C.REC"]
    labels = ["Si_Oxi", "SOI", "Si_Oxi_old"]
    names = ["Step", "Comment", "Duration (s)", "Pyrometer control", "Thermocouple control", "Power control", "",
             "Pyrometer setpoint [C]", "Thermocouple setpoint [C]", "Power setpoint [%]", "Temperature alarm",
             "Time base 1/10 s", "Gas flow band alarm", "Open vacuum valve", "Open purge valve", "Use secondary vacuum pump",
             "Descend Platen", "Top Center zone compensation", "Top Middle  zone compensation", "Top Edge zone compensation",
             "Bottom Center zone compensation", "Bottom Middle  zone compensation", "Bottom Edge zone compensation",
             "Vacuum setpoint", "Gas flow 1 [sccm]", "Gas flow 2 [sccm]", "Gas flow 3 [sccm]", "Gas flow 4 [sccm]", "Gas flow 5 [sccm]",
             "Gas flow 6 [sccm]", "Vacuum alarm", "Maximum vacuum rate"]
    
    # Interesting names: Duration, Pyrometer setpoint [C], Open vacuum valve, Open purge valve, Vacuume setpoint, Power setpoint [%]

    for l, label, c in zip(lists, labels, ["C0", "C1", "C2"]):
        file = folder / l
        df = pd.read_csv(file, skiprows=42, names=names, encoding='latin1', index_col=False)
        print(df)

        df["Periods (s)"] = df["Duration (s)"].cumsum()
        # ax.plot(df["Step"], df["Open vacuum valve"], label=f"Open vacuum valve - {label}", linestyle='-', marker='o', color=c)
        # ax.plot(df["Periods (s)"], df["Gas flow 1 [sccm]"], label=f"Open purge valve - {label}", linestyle='--', marker='o', color=c)
        # ax.plot(df["Periods (s)"], df["Open purge valve"], label=f"Open purge valve - {label}", linestyle='--', marker='o', color=c)
        # ax.plot(df["Periods (s)"], df["Pyrometer setpoint [C]"], label=f"Pyrometer setpoint - {label}", linestyle='-', marker='o', color=c)
        # ax.plot(df["Periods (s)"], df["Vacuum setpoint"], label=f"Vacuum setpoint - {label}", linestyle='--', marker='o', color=c)
        # ax.plot(df["Periods (s)"], df["Maximum vacuum rate"], label=f"Max. vaccum rate - {label}", linestyle='--', marker='o', color=c)

    
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (C)")
    # ax.set_xlim(0, 315)
    # ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    

if __name__ == "__main__":
    historical()
    # recipe()
    # x = (0, 100, 110, 140, 740)
    # y = (0, 1000, 1000, 700, 0)
    # plt.plot(x, y)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Temperature (C)")
    # plt.xlim(0, 300)
    # # ax.set_ylim(0, 100)
    # plt.grid(True, alpha=0.5)
    plt.show()