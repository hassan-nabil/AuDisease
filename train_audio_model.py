from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from audio_features import extract_audio_features_from_wav_file


AUDIO_ROOT = Path("AudioSample")
MODELS_DIR = Path("models")
AUDIO_MODEL_PATH = MODELS_DIR / "audio_pd_model.joblib"
AUDIO_SCALER_PATH = MODELS_DIR / "audio_pd_scaler.joblib"


def _iter_labeled_wavs() -> List[Tuple[Path, int]]:
    """
    Walk the AudioSample folder and return (path, label) pairs.

    Label convention:
        - 0: Healthy control (HC)
        - 1: Parkinson's disease (PD)

    We infer labels from directory names containing 'hc' or 'pd'.
    """
    pairs: List[Tuple[Path, int]] = []
    if not AUDIO_ROOT.exists():
        raise FileNotFoundError(
            f"Audio root folder '{AUDIO_ROOT}' not found. "
            "Place your WAV files under AudioSample/HC*/ and AudioSample/PD*/."
        )

    for wav_path in AUDIO_ROOT.rglob("*.wav"):
        parts = [p.lower() for p in wav_path.parts]

        label: int | None = None
        if any(p == "pd" or p.startswith("pd_") for p in parts):
            label = 1
        elif any(p == "hc" or p.startswith("hc_") for p in parts):
            label = 0

        if label is None:
            continue

        pairs.append((wav_path, label))

    if not pairs:
        raise RuntimeError(
            f"No labeled WAV files found under '{AUDIO_ROOT}'. "
            "Expected folders like 'HC', 'HC_AH', 'PD', 'PD_AH', etc."
        )

    return pairs


def main() -> None:
    """
    Train a simple PD vs HC classifier directly from AudioSample WAV files.

    Steps:
    - Traverse AudioSample and collect (path, label) pairs
    - Extract basic audio features for each file
    - Standardize features and train a RandomForestClassifier
    - Save the model and scaler to the models/ folder
    """
    MODELS_DIR.mkdir(exist_ok=True)

    pairs = _iter_labeled_wavs()
    print(f"Found {len(pairs)} labeled WAV files under '{AUDIO_ROOT}'.")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for path, label in pairs:
        try:
            feats = extract_audio_features_from_wav_file(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Skipping {path} due to error: {exc}")
            continue

        X_list.append(feats)
        y_list.append(label)

    if not X_list:
        raise RuntimeError("No features extracted; cannot train audio model.")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(train_X_scaled, train_y)

    preds = clf.predict(test_X_scaled)
    print("Classification report for audio-based PD vs HC model:")
    print(classification_report(test_y, preds))

    joblib.dump(clf, AUDIO_MODEL_PATH)
    joblib.dump(scaler, AUDIO_SCALER_PATH)
    print(f"Saved audio model to:   {AUDIO_MODEL_PATH}")
    print(f"Saved audio scaler to:  {AUDIO_SCALER_PATH}")


if __name__ == "__main__":
    main()


