from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
from scipy.signal import savgol_filter as sf
#---------------------------------------------------------------------------------

def _parse_prefixes(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _smooth_columns(df: pd.DataFrame,columns: list[str],window_length: int,polyorder: int,mode: str,) -> pd.DataFrame:
    for col in columns:
        series = df[col].to_numpy(dtype=float) #using numpy array to save columns values
        df[col] = sf(series,window_length=window_length,polyorder=polyorder,deriv=0,mode=mode,)
        # Acc and Gyr smoothing
    return df
#-----------------------------------------------------------------------------------------------------------------

# PARAMETERS DEF AND LOGIC FUNCTION CALLING DEF


def main() -> int:
    # folders and parameters definition
    project_root = Path(__file__).resolve().parents[2]
    input_dir = (project_root / "Data" / "processed" / "trimmed").resolve()
    output_dir = (project_root / "Data" / "processed" / "smoothed").resolve()
    window_length = 21
    polyorder = 3
    mode = "interp"
    include_prefixes = "acc_,gyr_"

    output_dir.mkdir(parents=True, exist_ok=True) #output folder creation

    csv_paths = sorted(input_dir.glob("*.csv")) # list of all trimmed data files' paths
    logging.info("Smoothing %d CSV files in %s", len(csv_paths), input_dir)
    # how many files found and where

    prefixes = _parse_prefixes(include_prefixes) # acc_ and gyr_
    written = 0 # for logging info
    skipped_no_cols = 0

    for csv_path in csv_paths: # each file iteration
        df = pd.read_csv(csv_path)
        output_path = output_dir / csv_path.name # creating file output direction

        data_columns = list(df.columns)
        smooth_cols = []
        for col in data_columns:
            if not any(col.startswith(prefix) for prefix in prefixes):
                continue # excluding other prefixes columns
            if col.startswith("ori_"):
                continue # excluding orientation columns
            smooth_cols.append(col)

        cols_to_smooth = smooth_cols
        if not cols_to_smooth: # if empty error otherwise continue
            logging.warning("No matching columns to smooth in %s.", csv_path.name)
            skipped_no_cols += 1
            df.to_csv(output_path, index=False)
            continue

        df = _smooth_columns(df, smooth_cols, window_length, polyorder, mode)
        # calling smooth function

        df.to_csv(output_path, index=False) # getting smooth values
        written += 1

    logging.info("Smoothed files written: %d | skipped no cols: %d",written,skipped_no_cols,
    )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
