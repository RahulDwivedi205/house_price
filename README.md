# India Housing Price Predictor

A machine learning web app that predicts house prices across India using Ridge Regression with polynomial features, powered by an AI Property Advisor using Groq (Llama 3.1).

## Overview

This project combines traditional ML with GenAI to predict housing prices in India. The model is trained on 250,000+ real housing listings. After getting a price prediction, users can chat with an AI advisor to get investment insights, price justification, and buying tips.

## Features

- Predict house prices instantly based on 19+ property parameters
- AI Property Advisor powered by Groq (Llama 3.1) - ask anything about the property
- Modern dark theme UI with glassmorphism design
- Interactive dataset exploration with charts
- Ridge Regression with polynomial features (R² = 0.85)

## Tech Stack

- Python
- Streamlit (web app)
- Scikit-learn (ML model)
- Groq API / Llama 3.1 (GenAI advisor)
- Plotly (charts)
- Pandas, NumPy

## Dataset

- 250,000 records
- 23 columns (39 after feature engineering)
- Covers 50+ cities across India
- Features: location, property type, size, amenities, floor, transport, etc.

## Model

**Ridge Regression with Polynomial Features**

- 16 engineered features (Size_Squared, Age_Squared, interaction terms, log transforms)
- Outlier removal using IQR method
- Feature selection: top 500 features (SelectKBest)
- Feature scaling: StandardScaler
- Polynomial interactions: degree 2
- Ridge regularization: alpha = 10.0
- R² = 0.85, MAE = 52 lakhs, RMSE = 68 lakhs

## Installation

### Prerequisites
- Python 3.8+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

1. Clone the repository:
```bash
git clone https://github.com/RahulDwivedi205/house_price.git
cd house_price
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

5. Run the app:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## How to Use

1. Fill in property details in the sidebar (location, BHK, size, amenities, etc.)
2. Click **Predict Price** to get the estimated price
3. Scroll down to **AI Property Advisor**
4. Type a question like *"Is this a good investment?"*
5. Click **Ask AI Advisor** to get an AI response

## Project Structure

```
house_price/
├── app.py                        # Streamlit web application
├── 01_eda.ipynb                  # Model training notebook
├── data/
│   └── india_housing_prices.csv  # Dataset
├── model_compressed.joblib       # Trained model (compressed)
├── linear_regression_model.joblib
├── scaler.joblib
├── selector.joblib
├── selected_features.joblib
├── requirements.txt
├── .env                          # API keys (not pushed to GitHub)
└── README.md
```

## Retrain the Model

1. Open `01_eda.ipynb` in Google Colab or Jupyter
2. Upload `data/india_housing_prices.csv`
3. Run all cells
4. Download the generated `.joblib` files

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com |

## Team

Built by **Team Charlie**

## Note

This is a machine learning model. Predictions are estimates based on historical data. Actual prices may vary based on current market conditions..
