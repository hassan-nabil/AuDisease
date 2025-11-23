from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from data_pipeline import prepare_data


MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "parkinsons_logreg.joblib"
SCALER_PATH = MODELS_DIR / "parkinsons_scaler.joblib"


def main() -> None:
    """
    Train a very simple baseline model for Parkinson's prediction and save it.

    This is intentionally small and easy to understand:
    - Load and standardize the dataset
    - Train a Logistic Regression classifier
    - Save both the model and the scaler to disk
    """
    MODELS_DIR.mkdir(exist_ok=True)

    train_X, test_X, train_y, test_y, scaler = prepare_data()

    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
    )
    model.fit(train_X, train_y)

    train_score = model.score(train_X, train_y)
    test_score = model.score(test_X, test_y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Baseline Logistic Regression model trained and saved.")
    print(f"Train accuracy: {train_score:.3f}")
    print(f"Test  accuracy: {test_score:.3f}")
    print(f"Model file : {MODEL_PATH}")
    print(f"Scaler file: {SCALER_PATH}")


if __name__ == "__main__":
    main()


