# Transformer Oil Degradation — Project Files

## Files Included

| File | Purpose |
|------|---------|
| `transformer_oil_dataset.csv` | Dataset with 51 samples + Remaining_Life column |
| `transformer_oil_ml_pipeline.py` | Complete ML pipeline (Steps 1–5) |
| `dashboard.py` | Streamlit interactive dashboard |

---

## Quick Start

### 1. Install dependencies
```
pip install pandas numpy scikit-learn matplotlib plotly streamlit
```

### 2. Run the ML pipeline (terminal output)
```
python transformer_oil_ml_pipeline.py
```

### 3. Run the dashboard (browser)
```
streamlit run dashboard.py
```
The dashboard opens at http://localhost:8501

---

## Important Note on Dataset Scale

Real transformer oil lasts **10–15 years** under normal operating conditions.
This dataset uses **100 simulation days** to represent accelerated aging,
which is standard practice in laboratory degradation studies.

To adapt for real-world use:
- Replace `Time_days` with `Time_months` (scale: 0 to 180 months)
- `Remaining_Life` would then be in months (0 to 180)
- All ML logic and formulas remain identical

---

## Project Pipeline Summary

```
Data Collection (EIS parameters)
        ↓
Data Preprocessing (Step 1)
        ↓
Exploratory Data Analysis (Step 2)
        ↓
Feature Engineering — Health Index, Remaining_Life (Step 3)
        ↓
Model 1: Random Forest Classifier → Good / Moderate / Bad (Step 4A)
Model 2: Random Forest Regressor  → Remaining Life in days (Step 4B)
        ↓
Prediction System (Step 5)
        ↓
Streamlit Dashboard (Step 6)
```

---

## Viva Answer: "Why 100 days and not 15 years?"

> "The dataset uses accelerated aging simulation — a standard technique
> where lab conditions speed up degradation to capture the full lifecycle
> within a short experiment. The EIS parameters (Rb, Rct, Impedance,
> Frequency Response) degrade in the same pattern regardless of timescale.
> In a real deployment, we would replace the time axis with months or years
> and recalibrate the remaining life target accordingly."
