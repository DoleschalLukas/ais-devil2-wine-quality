import pandas as pd
import numpy as np

INPUT_FILE = "data/winequality.parquet"

pd.read_parquet(INPUT_FILE)

def fetch_data():
    df = pd.read_parquet(INPUT_FILE)
    print("fetched data")
    print(df.head(5))


fetch_data()