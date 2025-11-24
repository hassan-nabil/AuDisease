# AuDisease
AuDisease is a small end‑to‑end prototype that explores **voice-based biomarkers** for helping with early screening for Parkinson’s Disease.

It combines two related pieces:

- A **tabular model** trained on the classic Max Little / UCI Parkinson’s voice dataset (`parkinsons.data`) that predicts Parkinson’s vs healthy from pre‑computed vocal features.
- An **audio model** trained directly on a curated `AudioSample` WAV dataset (healthy controls vs Parkinson’s) that estimates the probability a given recording belongs to a PD patient vs a healthy control.

> **Important:** This is a research / educational prototype, **not a medical device**. It must not be used for diagnosis or clinical decision‑making.

## 1. Problem & inspiration (for judges / non‑technical readers)

- **Problem:** Parkinson’s is often diagnosed late, after noticeable motor symptoms. Subtle voice changes can appear earlier, but are hard to detect by ear alone.
- **Inspiration:** Max Little’s research showed that **phone‑quality voice recordings** can be used to detect Parkinson’s with high accuracy by extracting carefully engineered features (jitter, shimmer, noise ratios, nonlinear dynamics, etc.).
- **Goal of this prototype:** Show an end‑to‑end experience where:
  - A user can explore how a model behaves on the **structured Parkinson’s dataset**.
  - A clinician or researcher can upload a **WAV recording** (from the AudioSample set or similar) and see the **audio model’s PD vs HC probability**.
  - The result is presented as an **estimated risk score** with clear caveats and explanations.

## 2. High‑level architecture

- **Frontend (static `index.html`)**
  - Modern, responsive single page.
  - Can:
    - Check backend health.
    - Display the list of input features the model expects.
    - Show a **demo prediction** using the tabular model.
    - Record a short voice sample in the browser (UX prototype only, not yet wired into the clinical audio model).
    - Let a user **upload a WAV file** and analyze it with the audio model.

- **Backend (FastAPI)**
  - `main.py` exposes:
    - `GET /` – serves the frontend.
    - `GET /health` – simple “is the API up?” check.
    - `GET /feature-names` – lists the ordered input features for the tabular model.
    - `GET /predict-demo` – runs the tabular model on a demo input derived from the dataset.
    - `POST /predict` – accepts a vector of numeric features and returns probability of Parkinson’s (tabular model).
    - `POST /predict-from-audio` – accepts a **WAV file**, extracts basic audio features, and returns a PD vs HC probability from the audio model.

- **Model & data (Python / scikit‑learn)**
  - Tabular dataset: `parkinsons.data` + description in `parkinsons.names` (UCI Parkinson’s voice dataset).
  - Tabular preprocessing & training:
    - `load_data.py` – quick check that the dataset loads and basic statistics look reasonable.
    - `data_pipeline.py` – reusable data loading, column cleaning, scaling and train/test split.
    - `train_baseline_model.py` – trains a simple **Logistic Regression** classifier and saves:
      - `models/parkinsons_logreg.joblib` – trained classifier.
      - `models/parkinsons_scaler.joblib` – feature scaler (StandardScaler).
  - Audio dataset: `AudioSample/` – WAV files for healthy controls (HC) and Parkinson’s patients (PD), plus demographics.
  - Audio feature extraction & training:
    - `audio_features.py` – extracts basic time‑ and frequency‑domain features from WAV waveforms.
    - `train_audio_model.py` – walks `AudioSample`, labels HC vs PD, trains a **RandomForest** classifier and saves:
      - `models/audio_pd_model.joblib` – trained audio classifier.
      - `models/audio_pd_scaler.joblib` – feature scaler for the audio features.

## 3. How to run the project (Windows, Python 3.11)

These steps assume you are in the `AuDisease` folder (where this README lives).

### 3.1 Create and activate the virtual environment

```powershell
py -3.11 -m venv .venv
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

### 3.4 Train and save the baseline tabular model

Train a simple Logistic Regression classifier on the structured Parkinson’s dataset:

```powershell
python train_baseline_model.py
```

This will create a `models` folder with:

- `parkinsons_logreg.joblib`
- `parkinsons_scaler.joblib`

and print the training and test accuracy.

### 3.5 (Optional) Train and save the audio model

If you have the `AudioSample` WAV dataset available locally:

```powershell
python train_audio_model.py
```

This will:

- Traverse `AudioSample/**` and label WAVs under HC*/ as healthy, PD*/ as Parkinson’s.
- Extract basic audio features from each WAV.
- Train a RandomForest PD vs HC classifier.
- Save:
  - `models/audio_pd_model.joblib`
  - `models/audio_pd_scaler.joblib`

### 3.6 Start the FastAPI backend

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
- **Record a short voice sample (UX prototype only)**
  - Click “Start recording” and grant microphone permission.
  - Speak for 5–10 seconds, then click “Stop recording”.
  - This is a UX preview; it does **not** currently feed into the audio model.
- **Analyze a WAV file with the audio model**
  - Use the “Analyze WAV file with audio model” section.
  - Choose a `.wav` file (for example from the `AudioSample` HC or PD folders).
  - Click “Analyze selected WAV”.
  - The frontend sends the file to `POST /predict-from-audio`, and the result updates the risk card using the audio PD vs HC model.

You can also explore all endpoints via the auto‑generated docs at `http://127.0.0.1:8000/docs`.

## 5. Current limitations & next steps

- **Audio model is trained on a specific curated dataset**
  - The PD vs HC audio classifier is trained on the local `AudioSample` WAV dataset only.
  - Generalization to other recording conditions, microphones or populations is unknown.

- **Features are generic, not identical to Max Little’s pipeline**
  - The audio model uses basic time/frequency statistics (duration, energy, zero‑crossings, simple spectral features), not the full jitter/shimmer/RPDE/PPE stack from the UCI tables.
  - Results should be treated as an exploratory signal, not a calibrated clinical score.

- **Browser recorder is not yet wired into the clinical audio model**
  - For reproducibility and fairness, the model currently expects WAV files similar to the training data.
  - The in‑browser recorder is there to show the intended future UX.

- **Planned next steps**
  - Experiment with richer audio features (e.g. pitch tracking, perturbation measures) that more closely match the tabular Parkinson’s datasets.
  - Evaluate calibration and robustness across recording conditions.
  - Iterate on UI to communicate uncertainty and “do not use for diagnosis” messaging more clearly.

## 6. Ethical and safety considerations

- This project is **not** a diagnostic tool and must not replace clinical assessment.
- Any deployment beyond a hackathon / datathon context would require:
  - Careful validation on real, diverse patient populations.
  - Regulatory review where applicable.
  - Clear communication with patients and clinicians about limitations and appropriate use.

## 7. Data sources & citations

- **UCI Parkinson’s voice dataset (tabular features)**
  - Based on the classic Max Little / UCI dataset of sustained phonations and derived vocal features.
  - Please see the UCI Machine Learning Repository entry for the Parkinson’s dataset for full attribution and licensing details.

- **Sustained “aaa” audio (HC_AH / PD_AH under `AudioSample/`)**
  - Source: open‑source Kaggle dataset by **Nutan Singh** (healthy controls and Parkinson’s patients producing sustained vowel phonations).
  - Used here for research and educational purposes; see the original Kaggle page for the exact license and terms of use.

- **Read text and spontaneous speech audio (`AudioSample/ReadText` and `AudioSample/SpontaneousDialogue`)**
  - Prior, Fred; Virmani, Tuhin; Iyer, Anu; Larson‑Prior, Linda; Kemp, Aaron; Rahmatallah, Yasir; et al. (2023).
    *Voice Samples for Patients with Parkinson’s Disease and Healthy Controls.* figshare. Dataset.
    [`https://doi.org/10.6084/m9.figshare.23849127.v1`](https://doi.org/10.6084/m9.figshare.23849127.v1)
  - Used in accordance with the dataset’s license; this project is strictly non‑clinical and for datathon / research demonstration only.
