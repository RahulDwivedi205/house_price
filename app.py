import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="India Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: #000000;
    background-image: 
        radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%),
        radial-gradient(at 0% 100%, rgba(236, 72, 153, 0.1) 0px, transparent 50%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a0a 0%, #111111 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.2);
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
}

section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa;
    font-weight: 700;
    text-shadow: 0 0 20px rgba(167, 139, 250, 0.3);
}

section[data-testid="stSidebar"] label {
    color: rgba(255, 255, 255, 0.85) !important;
    font-weight: 500;
    font-size: 0.95rem;
}

.stSelectbox, .stSlider, .stNumberInput, .stMultiSelect {
    color: #fff;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: rgba(20, 20, 20, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 10px;
    color: #fff !important;
}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover {
    border-color: rgba(139, 92, 246, 0.6) !important;
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.2);
}

.glass-card {
    background: linear-gradient(135deg, rgba(20, 20, 20, 0.9) 0%, rgba(30, 30, 30, 0.8) 100%);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 20px;
    padding: 32px 36px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(139, 92, 246, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    border-color: rgba(139, 92, 246, 0.5);
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 40%, #10b981 80%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.04em;
    margin-bottom: 8px;
    line-height: 1.1;
    text-shadow: 0 0 80px rgba(139, 92, 246, 0.5);
    animation: glow 3s ease-in-out infinite alternate;
}

@keyframes glow {
    from { filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.4)); }
    to { filter: drop-shadow(0 0 40px rgba(139, 92, 246, 0.8)); }
}

.hero-sub {
    font-size: 1.2rem;
    color: rgba(255, 255, 255, 0.6);
    font-weight: 400;
    margin-bottom: 32px;
    letter-spacing: 0.02em;
}

.metric-card {
    background: linear-gradient(135deg, rgba(15, 15, 15, 0.95) 0%, rgba(25, 25, 25, 0.9) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 18px;
    padding: 28px 26px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.1), transparent);
    transition: left 0.5s;
}

.metric-card:hover::before {
    left: 100%;
}

.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    border-color: rgba(139, 92, 246, 0.6);
    box-shadow: 0 12px 40px rgba(139, 92, 246, 0.3);
}

.metric-value {
    font-size: 1.85rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 8px 0 4px 0;
}

.metric-label {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

.price-result {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.25) 0%, rgba(59, 130, 246, 0.2) 100%);
    border: 2px solid rgba(139, 92, 246, 0.5);
    border-radius: 24px;
    padding: 48px 44px;
    text-align: center;
    animation: fadeSlideUp 0.7s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 20px 60px rgba(139, 92, 246, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.price-result::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
    animation: rotate 10s linear infinite;
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.price-amount {
    font-size: 3.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #c084fc, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 12px 0 6px 0;
    position: relative;
    z-index: 1;
    text-shadow: 0 0 40px rgba(192, 132, 252, 0.5);
}

.price-label {
    font-size: 1.05rem;
    color: rgba(255, 255, 255, 0.65);
    font-weight: 500;
    position: relative;
    z-index: 1;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 40px 0 16px 0;
    letter-spacing: -0.02em;
}

.gradient-divider {
    height: 3px;
    background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 50%, transparent 100%);
    border: none;
    margin: 20px 0 32px 0;
    border-radius: 3px;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.4);
}

.stButton > button {
    background: linear-gradient(135deg, #8b5cf6, #6366f1, #3b82f6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 18px 36px !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.03em;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100%;
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.4);
    text-transform: uppercase;
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.2);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.stButton > button:hover::before {
    width: 300px;
    height: 300px;
}

.stButton > button:hover {
    box-shadow: 0 12px 40px rgba(139, 92, 246, 0.6) !important;
    transform: translateY(-3px) scale(1.02);
}

.stExpander {
    background: rgba(20, 20, 20, 0.6) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: 16px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(20, 20, 20, 0.5);
    padding: 8px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: rgba(255, 255, 255, 0.6);
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    color: #fff !important;
}

.stDataFrame {
    background: rgba(15, 15, 15, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: 12px;
}

.footer-text {
    text-align: center;
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.3);
    margin-top: 60px;
    padding-bottom: 28px;
    font-weight: 300;
}
</style>
""", unsafe_allow_html=True)


BASE = Path(__file__).parent

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(BASE / "data" / "india_housing_prices.csv")

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(BASE / "model_compressed.joblib")

df = load_data()
model = load_model()
feature_names = list(model.feature_names_in_)

state_city_map = df.groupby("State")["City"].apply(lambda x: sorted(x.unique())).to_dict()
city_locality_map = df.groupby("City")["Locality"].apply(lambda x: sorted(x.unique())).to_dict()

ALL_AMENITIES = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]
with st.sidebar:
    st.markdown("## 🏠 Property Details")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📍 Location")
    state = st.selectbox("State", sorted(state_city_map.keys()), index=15)
    cities = state_city_map.get(state, [])
    city = st.selectbox("City", cities)
    localities = city_locality_map.get(city, [])
    locality = st.selectbox("Locality", localities)

    st.markdown("---")

    st.markdown("### 🏗️ Property Type")
    property_type = st.selectbox("Type", ["Apartment", "Independent House", "Villa"])
    bhk = st.slider("BHK", 1, 5, 2)
    size_sqft = st.slider("Size (SqFt)", 500, 5000, 1500, step=50)
    year_built = st.slider("Year Built", 1990, 2025, 2015)

    st.markdown("---")

    st.markdown("### 🏢 Building Details")
    furnished_status = st.selectbox("Furnished Status", ["Furnished", "Semi-furnished", "Unfurnished"])
    floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=5)
    total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=20)

    st.markdown("---")

    st.markdown("### 🏫 Neighbourhood")
    nearby_schools = st.slider("Nearby Schools", 1, 10, 5)
    nearby_hospitals = st.slider("Nearby Hospitals", 1, 10, 3)
    transport = st.selectbox("Public Transport", ["High", "Medium", "Low"])

    st.markdown("---")

    st.markdown("### ✨ Amenities & Extras")
    parking = st.selectbox("Parking Space", ["Yes", "No"])
    security = st.selectbox("Security", ["Yes", "No"])
    amenities_selected = st.multiselect("Amenities", ALL_AMENITIES, default=ALL_AMENITIES)
    facing = st.selectbox("Facing", ["North", "South", "East", "West"])
    owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
    availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
def build_features():
    current_year = 2026
    age = current_year - year_built

    median_ppsf = df["Price_per_SqFt"].median()

    row = {feat: 0 for feat in feature_names}

    row["Size_in_SqFt"] = size_sqft
    row["Price_per_SqFt"] = median_ppsf
    row["Year_Built"] = year_built
    row["Age_of_Property"] = age
    row["Nearby_Hospitals"] = nearby_hospitals

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
st.markdown('<div class="hero-title">🏠 India Housing Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Powered by Ridge Regression with polynomial features trained on 250,000+ real listings</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Location</div>
        <div class="metric-value">{city}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Property</div>
        <div class="metric-value">{bhk} BHK {property_type.split()[0]}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Size</div>
        <div class="metric-value">{size_sqft:,} sqft</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Built In</div>
        <div class="metric-value">{year_built}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")
if st.button("🔮  Predict Price"):
    with st.spinner("Crunching numbers …"):
        input_df = build_features()
        prediction = model.predict(input_df)[0]

    st.markdown(f"""
    <div class="price-result">
        <div class="price-label">Estimated Property Price</div>
        <div class="price-amount">₹ {prediction:,.2f} Lakhs</div>
        <div class="price-label" style="margin-top:6px;">≈ ₹ {prediction * 100_000:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 Your Input Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    summary_data = {
        "Parameter": [
            "State", "City", "Locality", "Property Type", "BHK",
            "Size (SqFt)", "Year Built", "Furnished", "Floor",
            "Total Floors", "Schools Nearby", "Hospitals Nearby",
            "Transport", "Parking", "Security", "Amenities",
            "Facing", "Owner Type", "Availability",
        ],
        "Value": [
            str(state), str(city), str(locality), str(property_type), str(bhk),
            f"{size_sqft:,}", str(year_built), str(furnished_status), str(floor_no),
            str(total_floors), str(nearby_schools), str(nearby_hospitals),
            str(transport), str(parking), str(security), ", ".join(amenities_selected),
            str(facing), str(owner_type), str(availability),
        ],
    }
    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True,
    )
st.markdown("")
st.markdown('<div class="section-title">📈 Dataset Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

with st.expander("🔍 Explore the training dataset", expanded=False):
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📋 Sample Data", "🗺️ Price by City"])

    with tab1:
        st.dataframe(df.describe().T.style.format("{:,.2f}"), use_container_width=True)

    with tab2:
        st.dataframe(df.head(100), use_container_width=True)

    with tab3:
        city_avg = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=True).reset_index()
        fig2 = px.bar(
            city_avg,
            x="Price_in_Lakhs",
            y="City",
            orientation="h",
            color="Price_in_Lakhs",
            color_continuous_scale=["#6366f1", "#8b5cf6", "#a78bfa", "#60a5fa"],
            labels={"Price_in_Lakhs": "Avg Price (₹ Lakhs)"},
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="rgba(255,255,255,0.85)",
            font_family="Poppins",
            coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=10),
            height=600,
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
<div class="footer-text">
    Built by Team Charlie with ❤️ India Housing Price Predictor
</div>
""", unsafe_allow_html=True)
