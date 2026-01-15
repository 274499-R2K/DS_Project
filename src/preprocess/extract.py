import logging # better message printing
import re # regex module to scan .zip filename and match eith .csv file
import zipfile # reading .zip
from pathlib import Path # path handling
from typing import Dict, Tuple
import pandas as pd


LOGGER = logging.getLogger(__name__) #module name
SENSORS = ["Accelerometer", "Gyroscope", "Orientation"] # list of what I keep


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_recording_id(zip_path: Path) -> Tuple[str, str]:
    # Parse class label and number from the zip filename stem.
    # Expected format: <class>_<number>-<date-time>
    # Example: wlk_01-2026-01-10_12-33-28

    stem = zip_path.stem
    match = re.match(r"^(?P<class>[A-Za-z]+)_(?P<num>\d+)-", stem)
    if not match:
        raise ValueError(f"Unrecognized zip name format: {stem}")
    class_label = match.group("class")
    number = match.group("num")
    return class_label, number


def _token_match(sensor: str, name: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(sensor)}(?![A-Za-z0-9])"
    return re.search(pattern, name, flags=re.IGNORECASE) is not None


def _sensor_match_score(sensor: str, member_name: str) -> Tuple[int, int, int]:
    lower_name = member_name.lower()
    lower_sensor = sensor.lower()
    is_csv = 0 if lower_name.endswith(".csv") else 1
    if _token_match(sensor, member_name):
        priority = 0
    elif lower_sensor in lower_name:
        priority = 1
    else:
        priority = 2
    return priority, is_csv, len(member_name)


def find_sensor_members(zf: zipfile.ZipFile) -> Dict[str, str]:
    # Find best matching zip members for the expected sensors.
    # Returns a mapping of sensor name to member name inside the zip.

    members = zf.namelist()
    results: Dict[str, str] = {}
    for sensor in SENSORS:
        candidates = []
        for member in members:
            score = _sensor_match_score(sensor, member)
            if score[0] < 2:
                candidates.append((score, member))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        results[sensor] = candidates[0][1]
    return results


def extract_and_load_all(
    raw_dir: Path | None = None,
) -> Dict[str, pd.DataFrame]:
    # Extract and load sensor CSVs from all zip files in raw_dir.
    root = _project_root()
    raw_dir = raw_dir or (root / "Data/raw")

    if not raw_dir.exists():
        LOGGER.warning("Raw directory does not exist: %s", raw_dir)
        return {}

    data: Dict[str, pd.DataFrame] = {}
    zip_files = sorted(raw_dir.glob("*.zip"))

    for zip_path in zip_files:
        LOGGER.info("Processing zip: %s", zip_path.name)
        class_label, number = parse_recording_id(zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            members = find_sensor_members(zf)

            missing = sorted(set(SENSORS) - members.keys())
            if missing:
                LOGGER.warning("Missing sensors in %s: %s", zip_path.name, ", ".join(missing))

            if not members:
                continue

            for sensor, member in members.items():
                with zf.open(member) as member_file:
                    df = pd.read_csv(member_file)
                key = f"{class_label}_{number}_{sensor}"
                data[key] = df
                LOGGER.info("Loaded %s: %s rows, %s cols", sensor, df.shape[0], df.shape[1])

    return data


if __name__ == "__main__":
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

