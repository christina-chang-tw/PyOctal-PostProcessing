from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def main():
    file = Path(r"E:\slot14.txt")
    df = pd.read_csv(file, sep="\t")
    df = df.drop(columns=["Absolute MSE"])

    output_file = Path(r"output.xlsx")

    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name='Slot 14', index=False)

    print(df)


if __name__ == "__main__":
    main()