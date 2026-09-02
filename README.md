# 🍃 Bangalore AQI & PM2.5 Prediction (XGBoost)

This repository contains a Machine Learning pipeline to analyze, clean, and predict Air Quality Index (AQI) levels—specifically focusing on **PM2.5** levels—in Bangalore using historical pollution data.

---

## 📌 Features

- **Data Cleaning & Preprocessing:** Detailed missing value handling, feature engineering, and outlier treatment (in `aqi_cleaned_data.ipynb`).
- **Feature Scaling:** Pre-fitted `scaler_ultra.pkl` standardizer for uniform feature bounds.
- **XGBoost Model (R² ~0.867):** Optimized XGBoost Regressor model saved as `best_pm25_xgb_ultra_0.8671.pkl`.
- **Application Interface:** Interactive Streamlit / FastAPI web app (`app.py`) for real-time predictions.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** XGBoost, Scikit-Learn
- **Visualization:** Matplotlib, Seaborn
- **Deployment/App:** Streamlit / Flask (`app.py`)

---

## 📂 Project Structure

```text
AQI-Banglore/
│── data/                              # Raw and cleaned dataset files
│── aqi_cleaned_data.ipynb             # Data preprocessing, EDA & model training
│── best_pm25_xgb_ultra_0.8671.pkl     # Trained XGBoost model file
│── scaler_ultra.pkl                   # Saved StandardScaler object
│── app.py                             # Web application / UI entry point
│── requirements.txt                   # List of Python dependencies
└── README.md                          # Project documentation
