#IMPORT AND CONSTANTS DEFINITION

import logging # better message printing
import re # regex module to scan .zip filename and match eith .csv file
import zipfile # reading .zip
from pathlib import Path # path handling
from typing import Dict, Tuple
import pandas as pd

LOGGER = logging.getLogger(__name__) #module name
SENSORS = ["Accelerometer", "Gyroscope", "Orientation"] # list of what I keep
#-------------------------------------------------------------------------

# PROJECT ROOT FINDER:

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]
# from current path (__file__) going up to layers to project folder

#----------------------------------------------------------------------------

# FILE NAME READING
# Returning class label and number from the zip filename stem.
# Expected format: <class>_<number>-<date-time>
# Example: wlk_01-2026-01-10_12-33-28

def parse_recording_id(zip_path: Path) -> Tuple[str, str]:
    stem = zip_path.stem # removing ".zip"
    match = re.match(r"^(?P<class>[A-Za-z]+)_(?P<num>\d+)-", stem)
    # r" for row string ; ^ start of the string ;
    # takes digit before underscore and put in <class>
    # takes number before dash sign and put in <num>

    if not match:
        raise ValueError(f"Unrecognized zip name format: {stem}") # zip file name incorrect

    # variable assignment and return
    class_label = match.group("class")
    number = match.group("num")
    return class_label, number
#-----------------------------------------------------------------------------------

# ZIP INNER FILES SELECTION
# Finding the wanted sensors zip members.
# Returns a mapping of sensor name to member name inside the zip.

def find_sensor_members(zf: zipfile.ZipFile) -> Dict[str, str]:
    members = zf.namelist() # list of all paths
    results: Dict[str, str] = {}  # dictionary initialization

    for s in SENSORS: # take only desired values
        target = f"{s}.csv".lower() #  take name with all lowercase
        candidates = []
        for m in members:
            if Path(m).name.lower() == target:  # take target sensors
                candidates.append(m)
        if candidates:
            results[s] = min(candidates, key=len) # choose the shortest
    return results
# returning for the current zip file the dict with: sensor_name : path
# ------------------------------------------------------------------------------------------------

# MAIN ZIP FOR LOOPING
# this is the function to called in the running process

def extract_and_load_all(raw_dir: Path | None = None,output_dir: Path | None = None,) -> Dict[str, pd.DataFrame]:

    # Extract and load sensor CSVs from all zip files in raw_dir.
    root = _project_root() # root folder
    raw_dir = raw_dir or (root / "Data/raw")  # passed raw dir or default if None

    if not raw_dir.exists():
        LOGGER.warning("Raw directory does not exist: %s", raw_dir)
        return {} # ERROR message if folder not existing

    output_dir = output_dir or (root / "Data" / "interim" / "extracted") # passed out dir or default
    output_dir.mkdir(parents=True, exist_ok=True) # folder creation or acceptance if existing

    data: Dict[str, pd.DataFrame] = {} # dataframe initialization
    zip_files = sorted(raw_dir.glob("*.zip")) # taking sorted zip files from input folder


    for zip_path in zip_files:  # main zip files loop
        LOGGER.info("Processing zip: %s", zip_path.name) # operation message
        class_label, number = parse_recording_id(zip_path) # extracting class and ID by proper func call

        with zipfile.ZipFile(zip_path) as zf:  # opening current file
            members = find_sensor_members(zf) # extract the desired sensor files with proper func call

            missing = sorted(set(SENSORS) - members.keys()) # compare desired vs obtained
            if missing:  # error if missing sensore existing (TRUE)
                LOGGER.warning("Missing sensors in %s: %s", zip_path.name, ", ".join(missing))

            if not members: # if members extracted are none (false) skip current zip loop cycle
                continue

            for sensor, member in members.items(): # key-value pair looping in dictionary
                with zf.open(member) as member_file: # opem path value file
                    df = pd.read_csv(member_file) # insert in dataframe

                out_path = output_dir / f"{class_label}_{number}_{sensor}.csv" # build/overwrite file in output folder
                df.to_csv(out_path, index=False) # df to csv

                key = f"{class_label}_{number}_{sensor}"
                data[key] = df # saving data also internally

                LOGGER.info("Loaded %s: %s rows, %s cols", sensor, df.shape[0], df.shape[1])

    return data
#-----------------------------------------------------------------------------------------------------------




# SCRIPT RUNNING SUMMARY

if __name__ == "__main__": # additional info showed when running as a script and not by external call
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_dir = _project_root() / "Data/raw"
    zip_files = sorted(raw_dir.glob("*.zip")) if raw_dir.exists() else []
    data = extract_and_load_all(raw_dir=raw_dir)

    print(f"Zips found: {len(zip_files)}")

    processed = {}
    for key, df in data.items():
        try:
            class_label, number, sensor = key.split("_", maxsplit=2)
        except ValueError:
            continue
        rec_id = f"{class_label}_{number}"
        processed.setdefault(rec_id, {})[sensor] = df

    print(f"Zips processed: {len(processed)}")

    for rec_id, sensors in sorted(processed.items()):
        parts = []
        for sensor_name in SENSORS:
            if sensor_name in sensors:
                df = sensors[sensor_name]
                parts.append(f"{sensor_name} ({df.shape[0]} rows)")
        sensors_summary = ", ".join(parts)
        print(f"{rec_id}: {sensors_summary}")

