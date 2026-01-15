from src.preprocess.extract_merge import extract_and_load_all

data = extract_and_load_all()
print(data.keys())  # all "<class>_<number>_<Sensor>" keys

df = data["lng_04_Gyroscope"]
print(df.head())