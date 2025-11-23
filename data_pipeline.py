import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("parkinsons.data")


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same column-cleaning logic used in the original notebook:
    - Replace spaces with underscores
    - Replace '(%)' with '_perc'
    - Replace '(dB)' with '_db'
    - Replace ':' with '_'
    - Lowercase everything
    - Remove any remaining '(...)' patterns
    """
    cols = [
        c.replace(" ", "_")
        .replace("(%)", "_perc")
        .replace("(dB)", "_db")
        .replace(":", "_")
        .lower()
        for c in df.columns
    ]
    cols = [re.sub(r"\((.+)\)", "", c) for c in cols]
    df = df.copy()
    df.columns = cols
    return df


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Parkinson's dataset from disk.

    Raises:
        FileNotFoundError: if the data file is missing.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dataset at '{DATA_PATH}'. "
            "Make sure 'parkinsons.data' is in the project folder."
        )
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_data(
    test_size: float = 0.3, random_state: int = 323
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Load, clean and split the data into train/test sets with standardized features.

    Returns:
        train_X, test_X, train_y, test_y, scaler
    """
    df = load_raw_data()
    df = _clean_column_names(df)

    # Drop the non-numeric 'name' column and keep 'status' as target.
    df_model = df.drop(columns=["name"])
    target_col = "status"

    feature_cols = [c for c in df_model.columns if c != target_col]

    X = df_model[feature_cols]
    y = df_model[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    train_X, test_X, train_y, test_y = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return train_X, test_X, train_y, test_y, scaler


if __name__ == "__main__":
    # Small, human-readable sanity check when run as a script.
    print("Loading and preparing Parkinson's dataset...")
    train_X, test_X, train_y, test_y, scaler = prepare_data()
    print(f"Training rows: {train_X.shape[0]}, Testing rows: {test_X.shape[0]}")
    print(f"Number of features: {train_X.shape[1]}")


