# IMPORT AND COSTANTS DEFINITION

import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

LOGGER = logging.getLogger(__name__) # to identify provenience of messages
SENSOR_PREFIX = {"Accelerometer": "acc_", "Gyroscope": "gyr_", "Orientation": "ori_"}
SENSORS = list(SENSOR_PREFIX.keys())
FREQ = "20ms"
# -----------------------------------------------------------------------------------------------
# ROOT FOLDER LOCATING
def project_root() -> Path:
    # Return the project root folder.
    return Path(__file__).resolve().parents[2]
#-----------------------------------------------------------------------------------------
# CREATING AND FILLING A NESTED DICTIONARIES
# Discover sensor CSV files and group them by their same recording key
# (group of 3 different sensors data of the same recording session aggregated)

def discover_files(extracted_dir: Path) -> Dict[str, Dict[str, Path]]:
    if not extracted_dir.exists(): # if not FALSE
        raise FileNotFoundError(f"Extracted folder not found: {extracted_dir}")

    grouped: Dict[str, Dict[str, Path]] = {}
    # creating a multidimensional nested dictionary:
    # {key 1 (rec_key name) : { key 2 (sensor) : associated path , ... }}


    for path in extracted_dir.rglob("*.csv"): # for every path in input folder
        parts = path.stem.rsplit("_", 2) # splitting path in 3: <class> <ID> <sensor>
        if len(parts) != 3:
            LOGGER.warning("Skipping unrecognized filename: %s", path.name)
            continue # iflen(parts) != 3 go on with next path

        class_label, rec_id, sensor = parts # assignment of name parts to variables
        rec_key = f"{class_label}_{rec_id}" # I'm taking future name rec_key

        grouped.setdefault(rec_key, {}) # adding the first <class>_<ID> key to grouped dict [1st layer]

        if sensor in grouped[rec_key]: # preventing having 2 same sensor keys
            continue

        grouped[rec_key][sensor] = path  # adding to each key pair the relative path [2nd layer]

    return grouped
#-------------------------------------------------------------------------------------------------------

# PREPARING EACH CSV FILE COLUMNS TO BE MERGED
# Load a sensor CSV from its path and return a dataframe with prefixed measurement columns
# and correct time and data format

def load_sensor_frame(path: Path, sensor: str) -> Optional[pd.DataFrame]:

    df = pd.read_csv(path) # read path associated csv

    new_cols = []
    for col in df.columns:
        new_cols.append(str(col).strip())
    df.columns = new_cols
    # conversion to string and space removing of columns names

    df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True)
    # convertion to pd dateformat with known ns unit and time zone assumed known

    drop_cols = [col for col in df.columns if col.lower() in {"second_elapsed", "seconds_elapsed"}]
    # extracting the "second_elapsed" columns
    if drop_cols:
        df = df.drop(columns=drop_cols) # dropping the columns

    measurement_cols = [col for col in df.columns if col != "time"]
    if measurement_cols:
        df[measurement_cols] = df[measurement_cols].apply(pd.to_numeric, errors="coerce")
    # ensuring columns with data to have numeric format

    prefix = SENSOR_PREFIX[sensor] # getting sensor prefix saved at the start
    rename = {col: f"{prefix}{col}" for col in measurement_cols}
    df = df.rename(columns=rename)
    # changing columns title name with sensor prefix + current name
    # from x -> acc_x . It's need to conserve the sensor belonging behaviour in the merged table

    return df
#-----------------------------------------------------------------------------------------------------

# FORCING RESAMPLE TO A COMMON MASTER TIMELINE
# all same record's sensor data need to have same interval and total number of time steps.

def resample_to_grid(df: pd.DataFrame, target_index: pd.DatetimeIndex, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
    # df -> input of one sensor table
    # target_index -> artificial master time axis
    # t_start and t_end -> overlap window external extremes

    df = df.sort_values("time").drop_duplicates(subset="time", keep="first") # sort and remove duplicate, unnecessary
    df = df.set_index("time") # setting time column as index needed by interpolate method

    df = df.loc[(df.index >= t_start) & (df.index <= t_end)] # cropping outer-window samples

    df = df.reindex(df.index.union(target_index)).sort_index()
    # I'm creating a new index incorporating the original timestamps indexes and the desired master timestamps series
    # Needed for interpolation since it is working interpolating with neighbours

    df = df.interpolate(method="time").ffill(limit=2).bfill(limit=2)
    # forward and backward filling up to 40ms to mitigate neighbour missing data
    df = df.reindex(target_index) # now I reduce to the master timestamps series
    return df
#---------------------------------------------------------------------------------------------------------------------------------

# MERGING
# Merge alligned sensors for a recording and save the merged as .csv

def merge_recording(rec_key: str, sensor_paths: Dict[str, Path], output_dir: Path) -> Optional[Path]:
    # rec_key class_ID pair key to access first layer
    # sensor_paths -> second layer's dictionary sensor:path -> I'm getting all sensor paths of rec_key
    # output_dir where to store merging

    frames = [] # list for each sensor's dataframe
    tmins = [] # each sensor’s first timestamp
    tmaxs = [] # each sensor’s last timestamp
    used_files = [] # names of files used for logging

    # we now loop to get
    for sensor in SENSORS:
        path = sensor_paths[sensor] # getting each sensor path
        df = load_sensor_frame(path, sensor) # loading in a df
        if df is None:
            return None
        frames.append(df)
        tmins.append(df["time"].min())
        tmaxs.append(df["time"].max())
        used_files.append(path.name)
        # storing data, time borders and path in dedicated lists

    t_start = max(tmins)
    t_end = min(tmaxs)
    target_index = pd.date_range(start=t_start, end=t_end, freq=FREQ)
    # definition of overlapping window and its inner timeline

    resampled = []
    for df in frames:
        resampled.append(resample_to_grid(df, target_index, t_start, t_end))
    # for each sensor I'm calling the resampling function appending every output to resampled list


    merged = pd.concat(resampled, axis=1)
    # horizontal merging of resampled frames of each sensor

    second_elapsed = (merged.index - merged.index[0]).total_seconds() # new time delta index converted in floatt
    merged.insert(0, "second_elapsed", second_elapsed) # inserting new index timeline
    merged = merged.reset_index(drop=True) # discarding the previous datetime index

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{rec_key}.csv"
    merged.to_csv(out_path, index=False)
    # saving as <class>_<ID>.csv
    return out_path
#-------------------------------------------------------------------------------------------------------------------

def run_merge(extracted_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> None:
    # Run the merge process for all recordings in extracted_dir.
    root = project_root()
    extracted_dir = extracted_dir or (root / "Data" / "interim" / "extracted")
    output_dir = output_dir or (root / "Data" / "interim" / "merged")
    # get all folder paths

    grouped = discover_files(extracted_dir)
    # call of the discovering files function and get the double layer dictionary

    merged_count = 0
    # iteration of rec_keys over all recordings in sorted order
    for rec_key, sensors in sorted(grouped.items()):
        missing = set(SENSORS) - sensors.keys() # check if some required sensor is missing
        if missing:
            continue
        out_path = merge_recording(rec_key, sensors, output_dir)
        if out_path is not None:
            merged_count += 1

    LOGGER.info("Recordings merged: %s", merged_count)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_merge()


if __name__ == "__main__":
    main()
