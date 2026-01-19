# IMPORT AND CONSTANTS DEFINITION

import logging
from pathlib import Path
import pandas as pd


LOGGER = logging.getLogger(__name__)

def project_root() -> Path:
    return Path(__file__).resolve().parents[2] # Return the project root folder.
#-----------------------------------------------------------------------------------------------------------------------------

# # Trim a dataframe based on elapsed seconds.

def trim_dataframe(df: pd.DataFrame, trim_start: float, trim_end: float, time_col: str = "second_elapsed") -> pd.DataFrame:
    #getting in input data df and window values
    # time_col = second_elapsed
    df = df.sort_values(time_col)
    t_min = df[time_col].min()
    t_max = df[time_col].max()

    # finding exact window's index values
    lower = t_min + trim_start
    upper = t_max - trim_end

    return df.loc[(df[time_col] >= lower) & (df[time_col] <= upper)].copy() #df filtering
#-----------------------------------------------------------------------------------------------------------

def process_file(in_path: Path, out_path: Path, overwrite: bool) -> bool:
    if out_path.exists() and not overwrite:
        LOGGER.info("Skipping %s (exists, use --overwrite to replace)", out_path.name)
        return False
    # getting data values and index
    df = pd.read_csv(in_path)
    df = df.sort_values("second_elapsed")
    t_min = df["second_elapsed"].min()
    t_max = df["second_elapsed"].max()

    trim_start = 3.0
    trim_end = 4.0
    trimmed = trim_dataframe(df, trim_start, trim_end) #function calling
    kept_rows = len(trimmed)
    if kept_rows == 0:
        LOGGER.warning("Skipping %s: empty after trim (t_max=%.3f)", in_path.name, t_max)
        return False

    #LOGGING SUMMARY
    kept_t_min = trimmed["second_elapsed"].min()
    kept_t_max = trimmed["second_elapsed"].max()
    original_rows = len(df)
    removed_rows = original_rows - kept_rows
    LOGGER.info("%s | rows %s -> %s (removed %s) | span %.3fs -> %.3fs",in_path.name,
                original_rows,kept_rows,removed_rows,(t_max - t_min),(kept_t_max - kept_t_min),)

    out_path.parent.mkdir(parents=True, exist_ok=True) # output saving
    trimmed.to_csv(out_path, index=False)
    return True
#--------------------------------------------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # folder description
    root = project_root()
    input_dir = root / "Data" / "interim" / "merged"
    output_dir = root / "Data" / "processed" / "trimmed"

    if not input_dir.exists(): # existence check
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    files = sorted(input_dir.glob("*.csv")) # input files existence check
    if not files:
        LOGGER.warning("No files found in %s with pattern *.csv", input_dir)
        return

    processed = 0
    for path in files:
        out_path = output_dir / path.name
        if process_file(path, out_path, False): #function call for each file
            processed += 1 # files count
    LOGGER.info("Trimmed files written: %s", processed)

if __name__ == "__main__":
    main()
