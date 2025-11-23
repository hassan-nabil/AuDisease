# AuDisease
This project uses voice-based biomarkers and machine learning to assist in early screening for Parkinson’s Disease. The system analyzes vocal features extracted from short audio samples and predicts the likelihood of Parkinson’s-related symptoms.

## Setup (very first step)

We will work inside a **virtual environment** so that this project’s Python packages stay separate from everything else on your computer.

### 1. Create and activate the virtual environment (Windows PowerShell)

From the `AuDisease` folder run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` at the beginning of your terminal prompt, which means the environment is active.

### 2. Install the required Python packages

With the virtual environment activated, install the dependencies listed in `requirements.txt`:

```powershell
pip install -r requirements.txt
```

At this point, the project is ready for the next step: adding a small script that loads the Parkinson’s dataset (`parkinsons.data`) and prepares a simple baseline machine learning model.