# AuDisease
AuDisease is a small end‑to‑end prototype that explores **voice-based biomarkers** for helping with early screening for Parkinson’s Disease.

A short voice sample is recorded in the browser, sent to a FastAPI backend, and run through a machine‑learning model trained on the classic **Max Little / UCI Parkinson’s voice dataset**. The current version uses the **tabular dataset only**; audio uploads are wired through to the backend but not yet transformed into the exact features used by the model.

> **Important:** This is a research / educational prototype, **not a medical device**. It must not be used for diagnosis or clinical decision‑making.

## 1. Problem & inspiration (for judges / non‑technical readers)

- **Problem:** Parkinson’s is often diagnosed late, after noticeable motor symptoms. Subtle voice changes can appear earlier, but are hard to detect by ear alone.
- **Inspiration:** Max Little’s research showed that **phone‑quality voice recordings** can be used to detect Parkinson’s with high accuracy by extracting carefully engineered features (jitter, shimmer, noise ratios, nonlinear dynamics, etc.).
- **Goal of this prototype:** Show an end‑to‑end experience where:
  - A user speaks into their browser.
  - The system connects that voice sample to a model trained on the Parkinson’s dataset.
  - The result is presented as an **estimated risk score** with clear caveats and explanations.

Right now, the model runs on the structured dataset; the audio path is implemented up to upload, and the next step is to extract comparable features from real audio.

## 2. High‑level architecture

- **Frontend (static `index.html`)**
  - Modern, responsive single page.
  - Can:
    - Check backend health.
    - Display the list of input features the model expects.
    - Show a **demo prediction** using the trained model.
    - Record a short voice sample in the browser and send it to the backend.

- **Backend (FastAPI)**
  - `main.py` exposes:
    - `GET /` – serves the frontend.
    - `GET /health` – simple “is the API up?” check.
    - `GET /feature-names` – lists the ordered input features for the model.
    - `GET /predict-demo` – runs the trained model on a demo input derived from the dataset.
    - `POST /predict` – accepts a vector of numeric features and returns probability of Parkinson’s.
    - `POST /predict-from-audio` – accepts an uploaded audio file (upload path proven; analysis TBD).

- **Model & data (Python / scikit‑learn)**
  - Dataset: `parkinsons.data` + description in `parkinsons.names` (UCI Parkinson’s voice dataset).
  - Preprocessing & training:
    - `load_data.py` – quick check that the dataset loads and basic statistics look reasonable.
    - `data_pipeline.py` – reusable data loading, column cleaning, scaling and train/test split.
    - `train_baseline_model.py` – trains a simple **Logistic Regression** classifier and saves:
      - `models/parkinsons_logreg.joblib` – trained classifier.
      - `models/parkinsons_scaler.joblib` – feature scaler (StandardScaler).

## 3. How to run the project (Windows, Python 3.11)

These steps assume you are in the `AuDisease` folder (where this README lives).

### 3.1 Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` at the beginning of your terminal prompt.

### 3.2 Install dependencies

```powershell
pip install -r requirements.txt
```

### 3.3 Quick data check (optional but recommended)

Confirm the Parkinson’s dataset file is readable and looks sensible:

```powershell
python load_data.py
```

You should see:

- Dataset shape (rows × columns).
- First few rows.
- Summary statistics (min, max, mean, etc.).

### 3.4 Train and save the baseline model

Train a simple Logistic Regression classifier on the structured Parkinson’s dataset:

```powershell
python train_baseline_model.py
```

This will create a `models` folder with:

- `parkinsons_logreg.joblib`
- `parkinsons_scaler.joblib`

and print the training and test accuracy.

### 3.5 Start the FastAPI backend

With the virtual environment still active:

```powershell
uvicorn main:app --reload
```

By default this serves the app at `http://127.0.0.1:8000`.

## 4. What you can try in the browser

Open `http://127.0.0.1:8000/` in your browser.

From the UI you can:

- **Check API health**
  - Click “Check API health” to call `GET /health`.
- **See model inputs**
  - Click “Show input features” to see the list of numeric input features expected by the model.
- **Run a demo prediction with the real model**
  - Click “Run demo with real model”.
  - The app calls `GET /predict-demo`, which uses the model and scaler to score a demo input built from the dataset.
- **Record and upload a short voice sample**
  - Click “Start recording” and grant microphone permission.
  - Speak for 5–10 seconds, then click “Stop & send to backend”.
  - The browser uploads the audio file to `POST /predict-from-audio`.
  - The backend confirms that the upload path is working (no analysis yet).

You can also explore all endpoints via the auto‑generated docs at `http://127.0.0.1:8000/docs`.

## 5. Current limitations & next steps

- **No true audio‑to‑feature pipeline yet**
  - The current model is trained on the tabular UCI dataset (features already computed).
  - Real audio recordings are uploaded, but the backend does not yet extract jitter, shimmer, NHR, RPDE, etc. from the waveform.

- **Planned next steps**
  - Add an audio feature extraction pipeline (e.g. using `librosa` and/or signal‑processing tools) to approximate the dataset’s vocal features from raw audio.
  - Train and evaluate a model directly on those extracted features.
  - Iterate on UI to communicate uncertainty, calibration and “do not use for diagnosis” messaging more clearly.

## 6. Ethical and safety considerations

- This project is **not** a diagnostic tool and must not replace clinical assessment.
- Any deployment beyond a hackathon / datathon context would require:
  - Careful validation on real, diverse patient populations.
  - Regulatory review where applicable.
  - Clear communication with patients and clinicians about limitations and appropriate use.
