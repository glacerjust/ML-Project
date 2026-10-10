import pandas as pd

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

def inspect():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    print("\nColumns & Types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\nTarget ('Churn') Distribution:")
    print(df["Churn"].value_counts(normalize=True))
    print("\nFirst 3 rows:")
    print(df.head(3))

if __name__ == "__main__":
    inspect()
