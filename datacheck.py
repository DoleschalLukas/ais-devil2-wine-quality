import pandas as pd
import numpy as np

INPUT_FILE = "data/winequality.parquet"

data = pd.read_parquet(INPUT_FILE)

print("info")
print(data.info())
print("describe")
print(data.describe())
print("null values")
print(data.isnull().sum())
