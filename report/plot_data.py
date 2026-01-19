
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TIME_COL = "second_elapsed"

ACC_COLS = ["acc_x", "acc_y", "acc_z"]
GYR_COLS = ["gyr_x", "gyr_y", "gyr_z"]
ORI_COLS = ["ori_qx","ori_qy", "ori_qz","ori_qw", "ori_roll","ori_pitch","ori_yaw",]

# WINDOW SIZE
start=5
end=25

LOGGER = logging.getLogger(__name__)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Error: CSV file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def plot_multiline(df: pd.DataFrame,x_col: str,y_cols: list[str],title: str,ylabel: str,ax: plt.Axes | None = None,) -> None:
    clean_df = df.dropna(subset=[x_col] + y_cols).sort_values(x_col)

    if ax is None:
        _, ax = plt.subplots()
    for col in y_cols:
        ax.plot(clean_df[x_col], clean_df[col], label=col)

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    if ax.figure:
        ax.figure.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-pipeline", action="store_true", default=False)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.run_pipeline:
        run_preprocess_pipeline()

    print("Inserire il nome del file di cui visualizzare i dati. ", "\n", " formato richiesto: classe_ID.csv","\n", "ES: wlk_01.csv")
    ans=input("Inserire il nome del file: ")
    script_dir = Path(__file__).resolve().parent
    my_list = ["trimmed","smoothed"]


    for target in my_list:
        csv_path = script_dir.parent / "Data" / "processed" / target / ans
        df = load_csv(csv_path)
        time_min = df[TIME_COL].min()
        time_max = df[TIME_COL].max()
        print(f"Loaded: {csv_path}")
        print(f"Rows: {len(df)}")
        print(f"Time range ({TIME_COL}): {time_min} to {time_max}")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
        fig.suptitle("Your title here", y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plot_multiline(df, TIME_COL, ACC_COLS, "Acceleration", "m/s^2", ax=axes[0])
        plot_multiline(df, TIME_COL, GYR_COLS, "Rotation Rate", "rad/s", ax=axes[1])
        plot_multiline(df, TIME_COL, ORI_COLS, "Orientation", "units", ax=axes[2])
        for ax in axes:
            ax.set_xlim(start, end)
        fig.tight_layout()
        fig.suptitle(f"{target}: {csv_path.name}", fontsize=12)

    plt.show()


if __name__ == "__main__":
    main()
