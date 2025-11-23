import pandas as pd


DATA_PATH = "parkinsons.data"


def main() -> None:
    """
    Simple helper script to confirm that the Parkinson's dataset
    can be loaded correctly.
    """
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded dataset from '{DATA_PATH}'.")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns.\n")

    print("First 5 rows:")
    print(df.head(), "\n")

    print("Summary statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()


