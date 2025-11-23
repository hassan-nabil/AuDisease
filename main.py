from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from audio_features import (
    extract_placeholder_features_from_audio_bytes,
    get_feature_columns,
)
from data_pipeline import _clean_column_names, load_raw_data


app = FastAPI(title="AuDisease – Parkinson's Screening API")


@app.get("/", response_class=HTMLResponse)
def root():
    """
    Serve the simple frontend located at static/index.html.
    """
    index_path = Path("static/index.html")
    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend file 'static/index.html' not found.",
        )
    return index_path.read_text(encoding="utf-8")


MODEL_PATH = Path("models/parkinsons_logreg.joblib")
SCALER_PATH = Path("models/parkinsons_scaler.joblib")


class FeaturesRequest(BaseModel):
    """
    For now, we expect the same numeric features that are in the dataset.

    Later, when we add audio processing, the frontend will send audio instead
    and the backend will convert it into these features automatically.
    """

    values: List[float]


def _load_model_and_scaler():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise RuntimeError(
            "Model or scaler file not found. "
            "Run 'python train_baseline_model.py' first."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@app.get("/health")
def health_check():
    """
    Simple endpoint so we can see if the API is running.
    """
    return {"status": "ok"}


@app.get("/feature-names")
def feature_names():
    """
    Expose the ordered list of feature names expected by the model.

    This helps the frontend (or a non-programmer) see exactly what the
    current model input looks like.
    """
    cols = get_feature_columns()
    return {"feature_names": cols}


@app.get("/predict-demo")
def predict_demo():
    """
    Run the trained model on a simple demo input derived from the dataset.

    We take the mean of each feature across the dataset as a single example
    input, then return the model's predicted probability. This lets the
    frontend exercise the real model without collecting audio yet.
    """
    df = load_raw_data()
    df = _clean_column_names(df)
    df = df.drop(columns=["name"])
    feature_cols = [c for c in df.columns if c != "status"]
    X = df[feature_cols]
    x_mean = X.mean().to_numpy().reshape(1, -1)

    model, scaler = _load_model_and_scaler()
    x_scaled = scaler.transform(x_mean)

    proba = model.predict_proba(x_scaled)[0, 1]
    label = int(proba >= 0.5)

    return {
        "predicted_label": label,
        "probability_parkinsons": float(proba),
        "note": "Demo prediction using the average feature vector from the dataset.",
    }


@app.post("/predict")
def predict(request: FeaturesRequest):
    """
    Predict Parkinson's status from a list of numeric features.

    Response is deliberately simple and human-friendly:
    - predicted_label: 0 (healthy) or 1 (Parkinson's)
    - probability_parkinsons: value between 0 and 1
    """
    model, scaler = _load_model_and_scaler()

    x = np.array(request.values, dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)

    proba = model.predict_proba(x_scaled)[0, 1]
    label = int(proba >= 0.5)

    return {
        "predicted_label": label,
        "probability_parkinsons": float(proba),
    }


@app.post("/predict-from-audio")
async def predict_from_audio(file: UploadFile = File(...)):
    """
    Experimental endpoint for audio-based prediction.

    Current behaviour:
    - Accepts an uploaded audio file (e.g. from the browser recorder)
    - Maps it to a placeholder feature vector based on the dataset
    - Runs the same Logistic Regression model and returns a probability

    NOTE: This does not yet implement real signal processing; it keeps the
    end-to-end path in place while we add proper audio features.
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Please upload an audio file.")

    raw_bytes = await file.read()

    features, feature_cols = extract_placeholder_features_from_audio_bytes(raw_bytes)

    model, scaler = _load_model_and_scaler()
    x_scaled = scaler.transform(features.reshape(1, -1))

    proba = model.predict_proba(x_scaled)[0, 1]
    label = int(proba >= 0.5)

    return {
        "predicted_label": label,
        "probability_parkinsons": float(proba),
        "feature_names": feature_cols,
        "note": "Experimental audio-based prediction using placeholder features "
        "derived from the structured dataset. Do not use for diagnosis.",
    }

