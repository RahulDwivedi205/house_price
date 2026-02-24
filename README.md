# 🏠 India Housing Price Predictor

A machine learning web application that predicts house prices across India using Random Forest Regressor algorithm.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

## 📋 Overview

This project uses machine learning to predict housing prices in India based on various features like location, property type, size, amenities, and more. The model is trained on 250,000+ real housing listings and achieves 98% accuracy.

## ✨ Features

- **Real-time Price Prediction**: Get instant price estimates for properties
- **Interactive Web Interface**: User-friendly Streamlit application
- **Comprehensive Inputs**: 19+ parameters including location, property details, and amenities
- **Visual Analytics**: Feature importance charts and data insights
- **Modern UI**: Dark theme with glassmorphism design
- **High Accuracy**: 98% R² score with Random Forest model

## 📊 Dataset

- **Size**: 250,000 records
- **Features**: 23 columns
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
**Random Forest Regressor**
- Ensemble learning method
- 100 decision trees
- Robust to overfitting
- Handles non-linear relationships

### Performance Metrics
- **R² Score**: 0.98 (98% accuracy)
- **Mean Absolute Error (MAE)**: 11.33 lakhs
- **Mean Squared Error (MSE)**: 471.63

### Training Process
1. Data loading and exploration
2. One-hot encoding for categorical variables
3. Train-test split (80-20)
4. Model training with Random Forest
5. Evaluation and validation
6. Model serialization

## 📁 Project Structure

```
gen_ai/
├── app.py                              # Streamlit web application
├── 01_eda.ipynb                        # Jupyter notebook for model training
├── data/
│   └── india_housing_prices.csv        # Dataset
├── model_compressed.joblib             # Trained model (compressed)
├── random_forest_regressor_model.joblib # Trained model
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

1. Open `01_eda.ipynb` in Jupyter Notebook
2. Run all cells sequentially
3. The trained model will be saved as `model_compressed.joblib`

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

Built with ❤️ using Streamlit & Scikit-learn

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This is a machine learning model and predictions are estimates based on historical data. Actual prices may vary based on market conditions and other factors not included in the model.
