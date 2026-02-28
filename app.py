# ==============================
# Import Required Libraries
# ==============================
import streamlit as st          # Streamlit for building web app
import pandas as pd             # Data manipulation
import numpy as np              # Numerical operations
import joblib                   # Load trained ML model
import plotly.express as px     # Interactive charts
import plotly.graph_objects as go
from pathlib import Path        # Handle file paths

# ==============================
# Streamlit Page Configuration
# ==============================
st.set_page_config(
    page_title="India Housing Price Predictor",  # Browser tab title
    page_icon="🏠",                               # Tab icon
    layout="wide",                               # Full width layout
    initial_sidebar_state="expanded",            # Sidebar open by default
)

# ==============================
# Custom CSS Styling
# ==============================
# Full UI Styling (Dark futuristic theme)
# NOTE: This section only handles frontend styling, not ML logic.
st.markdown("""
<style>
/* Entire CSS unchanged - Only styling */
...
</style>
""", unsafe_allow_html=True)


# ==============================
# Base Path for Files
# ==============================
BASE = Path(__file__).parent  # Gets current script directory


# ==============================
# Load Dataset (Cached)
# ==============================
# Cache prevents reloading file every time user interacts
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(BASE / "data" / "india_housing_prices.csv")


# ==============================
# Load Trained ML Model (Cached)
# ==============================
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(BASE / "model_compressed.joblib")


# Load data & model
df = load_data()
model = load_model()

# Extract feature names used during training
feature_names = list(model.feature_names_in_)


# ==============================
# Create Dropdown Mappings
# ==============================
# Mapping State → Cities
state_city_map = df.groupby("State")["City"].apply(lambda x: sorted(x.unique())).to_dict()

# Mapping City → Localities
city_locality_map = df.groupby("City")["Locality"].apply(lambda x: sorted(x.unique())).to_dict()

# Predefined Amenities List
ALL_AMENITIES = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]


# ==============================
# Sidebar Inputs
# ==============================
with st.sidebar:

    # Property Basic Details
    st.markdown("## 🏠 Property Details")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # -------- Location Section --------
    st.markdown("### 📍 Location")
    state = st.selectbox("State", sorted(state_city_map.keys()), index=15)
    cities = state_city_map.get(state, [])
    city = st.selectbox("City", cities)
    localities = city_locality_map.get(city, [])
    locality = st.selectbox("Locality", localities)

    st.markdown("---")

    # -------- Property Type Section --------
    st.markdown("### 🏗️ Property Type")
    property_type = st.selectbox("Type", ["Apartment", "Independent House", "Villa"])
    bhk = st.slider("BHK", 1, 5, 2)
    size_sqft = st.slider("Size (SqFt)", 500, 5000, 1500, step=50)
    year_built = st.slider("Year Built", 1990, 2025, 2015)

    st.markdown("---")

    # -------- Building Details --------
    st.markdown("### 🏢 Building Details")
    furnished_status = st.selectbox("Furnished Status", ["Furnished", "Semi-furnished", "Unfurnished"])
    floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=5)
    total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=20)

    st.markdown("---")

    # -------- Neighbourhood Info --------
    st.markdown("### 🏫 Neighbourhood")
    nearby_schools = st.slider("Nearby Schools", 1, 10, 5)
    nearby_hospitals = st.slider("Nearby Hospitals", 1, 10, 3)
    transport = st.selectbox("Public Transport", ["High", "Medium", "Low"])

    st.markdown("---")

    # -------- Amenities --------
    st.markdown("### ✨ Amenities & Extras")
    parking = st.selectbox("Parking Space", ["Yes", "No"])
    security = st.selectbox("Security", ["Yes", "No"])
    amenities_selected = st.multiselect("Amenities", ALL_AMENITIES, default=ALL_AMENITIES)
    facing = st.selectbox("Facing", ["North", "South", "East", "West"])
    owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
    availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])


# ==============================
# Feature Engineering Function
# ==============================
def build_features():
    """
    Converts user inputs into a dataframe
    that matches model training features.
    Handles one-hot encoding manually.
    """

    current_year = 2026
    age = current_year - year_built  # Derived feature

    median_ppsf = df["Price_per_SqFt"].median()

    # Initialize all features to 0
    row = {feat: 0 for feat in feature_names}

    # Numerical Features
    row["Size_in_SqFt"] = size_sqft
    row["Price_per_SqFt"] = median_ppsf
    row["Year_Built"] = year_built
    row["Age_of_Property"] = age
    row["Nearby_Hospitals"] = nearby_hospitals

    # One-hot encoding for categorical variables
    key_state = f"State_{state}"
    if key_state in row:
        row[key_state] = 1

    key_city = f"City_{city}"
    if key_city in row:
        row[key_city] = 1

    key_loc = f"Locality_{locality}"
    if key_loc in row:
        row[key_loc] = 1

    key_pt = f"Property_Type_{property_type}"
    if key_pt in row:
        row[key_pt] = 1

    key_tr = f"Public_Transport_Accessibility_{transport}"
    if key_tr in row:
        row[key_tr] = 1

    key_sec = f"Security_{security}"
    if key_sec in row:
        row[key_sec] = 1

    key_av = f"Availability_Status_{availability}"
    if key_av in row:
        row[key_av] = 1

    # Amenities encoding
    amenities_str = ", ".join(sorted(amenities_selected))
    key_am = f"Amenities_{amenities_str}"
    if key_am in row:
        row[key_am] = 1
    else:
        for f in feature_names:
            if f.startswith("Amenities_"):
                stored_set = set(a.strip() for a in f.replace("Amenities_", "").split(","))
                if stored_set == set(amenities_selected):
                    row[f] = 1
                    break

    return pd.DataFrame([row], columns=feature_names)


# ==============================
# Main UI Section
# ==============================
st.markdown('<div class="hero-title">🏠 India Housing Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Powered by Ridge Regression with polynomial features trained on 250,000+ real listings</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ==============================
# Top Summary Cards
# ==============================
col1, col2, col3, col4 = st.columns(4)

# Display selected inputs dynamically
with col1:
    st.markdown(f"""...""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""...""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""...""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""...""", unsafe_allow_html=True)


# ==============================
# Prediction Button Logic
# ==============================
if st.button("🔮  Predict Price"):

    # Show loading spinner
    with st.spinner("Crunching numbers …"):
        input_df = build_features()
        prediction = model.predict(input_df)[0]

    # Display Prediction Result
    st.markdown(f"""...""", unsafe_allow_html=True)

    # Display Input Summary Table
    st.dataframe(pd.DataFrame(summary_data),
                 use_container_width=True,
                 hide_index=True)


# ==============================
# Dataset Exploration Section
# ==============================
# Allows user to inspect training data

with st.expander("🔍 Explore the training dataset", expanded=False):

    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📋 Sample Data", "🗺️ Price by City"])

    # Descriptive statistics
    with tab1:
        st.dataframe(df.describe().T.style.format("{:,.2f}"), use_container_width=True)

    # Raw sample rows
    with tab2:
        st.dataframe(df.head(100), use_container_width=True)

    # City-wise average price chart
    with tab3:
        city_avg = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=True).reset_index()
        fig2 = px.bar(...)
        st.plotly_chart(fig2, use_container_width=True)


# ==============================
# Footer
# ==============================
st.markdown("""
<div class="footer-text">
    Built by Team Charlie with ❤️ India Housing Price Predictor
</div>
""", unsafe_allow_html=True)