# 🇮🇳 AI-Based Economic Growth Forecasting

An interactive machine learning project designed to explore India's economic growth trends and generate GDP growth forecasts using historical economic indicators.

## 📌 Project Overview

Economic growth forecasting can help researchers and students understand historical economic patterns and explore possible future trends.

This project uses historical Indian economic data and machine learning techniques to build an interactive forecasting dashboard using Streamlit.

## 🎯 Objectives

- Analyze India's historical GDP growth.
- Explore economic indicators associated with growth.
- Apply machine learning to economic forecasting.
- Visualize historical data and forecast results.
- Provide an interactive dashboard for exploring the results.

## ✨ Features

- Interactive Streamlit dashboard.
- Historical economic data visualization.
- GDP growth forecasting interface.
- Economic indicators such as inflation, unemployment, exports, imports, and capital formation.
- Model evaluation and comparison.
- Forecast year selection.

## 🛠️ Technologies Used

- **Python** — data processing and model development
- **Pandas & NumPy** — data manipulation
- **Scikit-learn** — model evaluation and machine learning utilities
- **XGBoost** — forecasting model
- **Matplotlib / Plotly** — data visualization
- **Streamlit** — interactive web dashboard
- **Jupyter Notebook** — data collection and experimentation
- **Git & GitHub** — version control

## 📂 Project Structure

```text
economic-growth-forecasting/
├── dashboard/
│   └── app.py
├── data/
│   └── processed/
├── models/
│   └── best_model.joblib
├── notebooks/
│   └── 01_data_collection.ipynb
└── .gitignore
```

## 📊 Dataset

The project uses historical economic indicators for India, covering data from 1991 to 2025 where available.

The indicators include:

- GDP growth
- Inflation
- Unemployment
- Exports as a percentage of GDP
- Imports as a percentage of GDP
- Gross capital formation as a percentage of GDP

World Bank indicator codes are used in the data collection workflow.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Amit-kumar-vishwakarma/economic-growth-forecasting.git
cd economic-growth-forecasting
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

Install the packages used by the project:

```bash
pip install streamlit pandas numpy scikit-learn xgboost matplotlib plotly joblib
```

### 5. Run the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL displayed in the terminal to access the dashboard.

## 🤖 Machine Learning Approach

The project explores XGBoost for GDP growth forecasting and compares its performance with baseline and other regression models.

Historical observations and lagged economic indicators are used to construct model inputs.

Model performance is evaluated using metrics such as:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## ⚠️ Limitations

Economic forecasting is challenging, particularly when the available dataset contains a limited number of annual observations.

Model predictions may be inaccurate and should not be interpreted as official economic forecasts. Some future indicator values shown in the dashboard may be illustrative rather than independently predicted by trained models.

The results are intended for educational and exploratory purposes, not financial or policy-making decisions.

## 🚀 Future Improvements

- Expand the historical dataset.
- Improve time-series validation and model selection.
- Explore dedicated time-series forecasting methods.
- Add more economic indicators and country comparisons.
- Improve forecast uncertainty estimation.
- Deploy the dashboard for public access.

## 🎓 Project Purpose

Developed as an academic minor project to explore data analysis, machine learning, economic forecasting, and interactive dashboard development.

## 📄 License

No license has been specified yet. All rights and reuse permissions remain with the repository owner unless a license is added.

---

**Note:** This is an educational project. Forecasts are experimental and are not official economic projections.
