from typing import List, Tuple

import numpy as np

from data_pipeline import _clean_column_names, load_raw_data


def get_feature_columns() -> List[str]:
    """
    Return the ordered list of feature column names used by the model.

    This mirrors the logic used in the training pipeline: drop the 'name'
    column and keep everything except the 'status' target.
    """
    df = load_raw_data()
    df = _clean_column_names(df)
    df = df.drop(columns=["name"])
    cols = [c for c in df.columns if c != "status"]
    return cols


def extract_placeholder_features_from_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, List[str]]:
    """
    TEMPORARY STUB: map an uploaded audio file into a model input vector.

    For now, this does NOT do real signal processing. Instead, it:
    - Reads the Parkinson's dataset
    - Computes the mean of each feature column
    - Returns that mean feature vector

    This keeps the end-to-end plumbing (upload -> features -> model) in place
    while we design and implement a proper audio feature extraction pipeline.
    """
    # NOTE: audio_bytes is currently unused; kept here to document the future API.
    del audio_bytes

    df = load_raw_data()
    df = _clean_column_names(df)
    df = df.drop(columns=["name"])

    feature_cols = [c for c in df.columns if c != "status"]
    X = df[feature_cols]
    x_mean = X.mean().to_numpy()

    return x_mean, feature_cols



