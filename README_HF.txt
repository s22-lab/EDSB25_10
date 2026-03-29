# HR Attrition Prediction – Gradio App

This Space hosts an HR attrition prediction demo using:
- Logistic Regression (conservative, higher precision)
- Decision Tree (aggressive, higher recall)

## How it works

- Input: 12 raw HR features (satisfaction, experience, department, role, travel, marital status, overtime)
- Preprocessing:
  - Median imputation
  - Log transforms for tenure features
  - Scaling (StandardScaler)
  - One‑hot encoding
  - 18‑feature selection (after VIF analysis)
- Output: Attrition probability, prediction (Stay/Leave), risk level, and recommended action.

## Files

- `app.py` – Gradio interface and prediction logic
- `preprocessing.py` – Training-time and inference-time preprocessing
- `models/` – Saved models and preprocessing artifacts
  - `logistic_regression_best.pkl`
  - `dt_attrition_model.joblib`
  - `scaler.pkl`
  - `imputer.pkl`
  - `logreg_model_metadata.json`
  - `dt_model_metadata.json`
- `requirements.txt` – Dependencies tested with Python 3.10–3.12

## Usage

Single employee:
- Fill sliders and dropdowns
- Choose model: Logistic Regression (Conservative) or Decision Tree (Aggressive)
- Click **Predict Attrition Risk**

Batch:
- Upload CSV with 12 raw columns (see “Model Information” tab in the app)
- Click **Predict All Employees**

## Notes

- Models are trained on a cleaned subset of the IBM HR Attrition dataset.
- Thresholds are recall‑optimized to prioritize catching potential leavers.
