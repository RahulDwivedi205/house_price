# 🏠 India Housing Price Predictor

A machine learning web application that predicts house prices across India using Ridge Regression with polynomial features.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

## 📋 Overview

This project uses machine learning to predict housing prices in India based on various features like location, property type, size, amenities, and more. The model is trained on 250,000+ real housing listings using Ridge Regression with polynomial features for improved accuracy.

## ✨ Features

- **High Accuracy**: Get instant price estimates for properties
- **Interactive Web Interface**: User-friendly Streamlit application
- **Comprehensive Inputs**: 19+ parameters including location, property details, and amenities
- **Visual Analytics**: Feature importance charts and data insights
- **Modern UI**: Dark theme with glassmorphism design
- **Advanced Model**: Ridge Regression with polynomial features and comprehensive preprocessing

## 📊 Dataset

- **Size**: 250,000 records
- **Features**: 23 columns (39 after feature engineering)
- **Coverage**: Major cities across India
- **Attributes**:
  - Location: State, City, Locality
  - Property: Type, BHK, Size (SqFt), Year Built
  - Pricing: Price in Lakhs, Price per SqFt
  - Building: Floor Number, Total Floors, Furnished Status
  - Neighborhood: Nearby Schools, Hospitals, Public Transport
  - Amenities: Playground, Gym, Garden, Pool, Clubhouse
  - Additional: Parking, Security, Facing, Owner Type, Availability

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd gen_ai
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8502`

### Using the Application

1. **Select Location**: Choose State, City, and Locality
2. **Property Details**: Specify property type, BHK, size, and year built
3. **Building Information**: Enter floor details and furnished status
4. **Neighborhood**: Add nearby facilities information
5. **Amenities**: Select available amenities
6. **Predict**: Click the "Predict Price" button to get the estimate

## 🤖 Model Information

### Algorithm
**Ridge Regression with Polynomial Features**
- Advanced feature engineering (16 new features including polynomial terms)
- Outlier removal using IQR method
- Feature selection (top 500 features)
- Feature scaling (StandardScaler)
- Polynomial feature interactions (degree 2)
- Ridge regularization (alpha=10.0)

### Performance Metrics
Run the `01_eda.ipynb` notebook to train the model and see the latest performance metrics.

### Training Process
1. Data loading and exploration
2. Advanced feature engineering (16 new features: Age, Price_per_BHK, Floor_Ratio, Size_Squared, Size_Cubed, Log transformations, etc.)
3. Outlier removal using IQR method
4. One-hot encoding for categorical variables
5. Feature selection (SelectKBest - top 500 features)
6. Feature scaling (StandardScaler)
7. Polynomial feature creation (degree 2, interaction terms)
8. Train-test split (80-20)
9. Model training with Ridge Regression (alpha=10.0)
10. Evaluation and validation
11. Model serialization

## 📁 Project Structure

```
gen_ai/
├── app.py                              # Streamlit web application
├── 01_eda.ipynb                        # Jupyter notebook for model training
├── data/
│   └── india_housing_prices.csv        # Dataset
├── linear_regression_model.joblib      # Trained Ridge model
├── model_compressed.joblib             # Compressed model (for app)
├── scaler.joblib                       # Feature scaler
├── selector.joblib                     # Feature selector
├── poly.joblib                         # Polynomial transformer
├── selected_features.joblib            # Selected feature names
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Joblib**: Model serialization

## 📈 Model Training

To retrain the model:

1. Open `01_eda.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. The trained model will be saved as:
   - `linear_regression_model.joblib`
   - `model_compressed.joblib`
   - `scaler.joblib`
   - `selector.joblib`
   - `poly.joblib`
   - `selected_features.joblib`

## 🎨 UI Features

- **Dark Theme**: Modern black background with gradient overlays
- **Glassmorphism**: Translucent cards with blur effects
- **Responsive Design**: Works on desktop and mobile
- **Interactive Elements**: Hover effects and animations
- **Data Visualization**: Feature importance charts
- **Input Summary**: Detailed parameter display

## 🔮 Future Enhancements

- [ ] Add more cities and localities
- [ ] Implement price trend analysis
- [ ] Add comparison with market rates
- [ ] Include property images
- [ ] Add user authentication
- [ ] Deploy on cloud platform
- [ ] Add API endpoints
- [ ] Implement caching for faster predictions

## 📝 License

This project is open source and available for educational purposes.

## 👥 Author

Built by Team Charlie with ❤️ · India Housing Price Predictor

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This is a machine learning model and predictions are estimates based on historical data. Actual prices may vary based on market conditions and other factors not included in the model.
