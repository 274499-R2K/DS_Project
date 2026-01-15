from src.preprocess.extract import extract_and_load_all

data = extract_and_load_all()
print(data.keys())  # all "<class>_<number>_<Sensor>" keys

df = data["srt_03_Accelerometer"]
print(df.head())
print(df.tail())
