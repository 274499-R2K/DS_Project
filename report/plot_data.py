
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TIME_COL = "second_elapsed"

ACC_COLS = ["acc_x", "acc_y", "acc_z"]
GYR_COLS = ["gyr_x", "gyr_y", "gyr_z"]
ORI_COLS = ["ori_qx","ori_qy", "ori_qz","ori_qw", "ori_roll","ori_pitch","ori_yaw",]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Error: CSV file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def plot_multiline(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    ylabel: str,
) -> None:
    clean_df = df.dropna(subset=[x_col] + y_cols).sort_values(x_col)

    plt.figure()
    for col in y_cols:
        plt.plot(clean_df[x_col], clean_df[col], label=col)

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir.parent / "Data" / "interim" / "merged" / ("srt_05.csv")

    df = load_csv(csv_path)

    time_min = df[TIME_COL].min()
    time_max = df[TIME_COL].max()
    print(f"Loaded: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Time range ({TIME_COL}): {time_min} to {time_max}")

    plot_multiline(df, TIME_COL, ACC_COLS, "Acceleration", "m/s^2")
    plot_multiline(df, TIME_COL, GYR_COLS, "Rotation Rate", "rad/s")
    plot_multiline(df, TIME_COL, ORI_COLS, "Orientation", "units")

    plt.show()


if __name__ == "__main__":
    main()
