# India Housing Price Predictor

A machine learning web app that predicts house prices across India using Ridge Regression with polynomial features.

## Overview

This project uses machine learning to predict housing prices in India. The model is trained on 250,000+ real housing listings and considers features like location, property type, size, amenities, and more.

## Features

- Get instant price estimates for properties
- User-friendly Streamlit web interface
- 19+ input parameters
- Modern dark theme UI
- Ridge Regression with polynomial features

## Dataset

- 250,000 records
- 23 columns (39 after feature engineering)
- Covers major cities across India
- Includes location, property details, pricing, amenities, etc.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd gen_ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the web app:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8502`

### How to use:

1. Select location (State, City, Locality)
2. Enter property details (type, BHK, size, year)
3. Add building info (floors, furnished status)
4. Select amenities
5. Click "Predict Price"

## Model Information

### Algorithm
Ridge Regression with Polynomial Features

What we did:
- Feature engineering (16 new features)
- Outlier removal (IQR method)
- Feature selection (top 500 features)
- Feature scaling (StandardScaler)
- Polynomial interactions (degree 2)
- Ridge regularization (alpha=10.0)

### Performance
- R² Score: 0.85 (85% variance explained)
- MAE: 52 lakhs
- RMSE: 68 lakhs

### Training Process
1. Load and explore data
2. Create 16 engineered features
3. Remove outliers
4. One-hot encode categorical variables
5. Select top 500 features
6. Scale features
7. Create polynomial features
8. Train Ridge model
9. Evaluate and save

## Project Structure

```
gen_ai/
├── app.py                              # Web application
├── 01_eda.ipynb                        # Model training notebook
├── data/
│   └── india_housing_prices.csv        # Dataset
├── *.joblib                            # Model files
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```

## Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Model Training

To retrain:

1. Open `01_eda.ipynb` in Jupyter or Google Colab
2. Run all cells
3. Model files will be saved automatically

## Future Ideas

- Add more cities
- Price trend analysis
- Market comparison
- Property images
- API endpoints

## Team

Built by Team Charlie

## Note

This is a machine learning model. Predictions are estimates based on historical data. Actual prices may vary.
