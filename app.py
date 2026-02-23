import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="India Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS – dark glassmorphism theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- Main background ---------- */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #a78bfa;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* ---------- Glass card ---------- */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 28px 32px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    margin-bottom: 20px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(124, 58, 237, 0.12);
}

/* ---------- Hero ---------- */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
    margin-bottom: 4px;
    line-height: 1.15;
}
.hero-sub {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.5);
    font-weight: 400;
    margin-bottom: 28px;
}

/* ---------- Metric card ---------- */
.metric-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 22px 24px;
    text-align: center;
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #a78bfa;
    margin: 6px 0 2px 0;
}
.metric-label {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

/* ---------- Price result ---------- */
.price-result {
    background: linear-gradient(135deg, rgba(124,58,237,0.18) 0%, rgba(96,165,250,0.12) 100%);
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 20px;
    padding: 36px 40px;
    text-align: center;
    animation: fadeSlideUp 0.6s ease;
}
.price-amount {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c084fc, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 8px 0 4px 0;
}
.price-label {
    font-size: 1rem;
    color: rgba(255,255,255,0.55);
    font-weight: 500;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---------- Section title ---------- */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 32px 0 12px 0;
    letter-spacing: -0.01em;
}

/* ---------- Divider ---------- */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 50%, transparent 100%);
    border: none;
    margin: 16px 0 28px 0;
    border-radius: 2px;
}

/* ---------- Button override ---------- */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.02em;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.4) !important;
    transform: translateY(-2px);
}

/* ---------- Footer ---------- */
.footer-text {
    text-align: center;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.25);
    margin-top: 48px;
    padding-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load data & model (cached)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Helpers – mapping for cascading dropdowns
# ─────────────────────────────────────────────
state_city_map = df.groupby("State")["City"].apply(lambda x: sorted(x.unique())).to_dict()
city_locality_map = df.groupby("City")["Locality"].apply(lambda x: sorted(x.unique())).to_dict()

ALL_AMENITIES = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]


# ─────────────────────────────────────────────
# Sidebar inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Property Details")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ── Location ──
    st.markdown("### 📍 Location")
    state = st.selectbox("State", sorted(state_city_map.keys()), index=15)
    cities = state_city_map.get(state, [])
    city = st.selectbox("City", cities)
    localities = city_locality_map.get(city, [])
    locality = st.selectbox("Locality", localities)

    st.markdown("---")

    # ── Property Type ──
    st.markdown("### 🏗️ Property Type")
    property_type = st.selectbox("Type", ["Apartment", "Independent House", "Villa"])
    bhk = st.slider("BHK", 1, 5, 2)
    size_sqft = st.slider("Size (SqFt)", 500, 5000, 1500, step=50)
    year_built = st.slider("Year Built", 1990, 2025, 2015)

    st.markdown("---")

    # ── Building Details ──
    st.markdown("### 🏢 Building Details")
    furnished_status = st.selectbox("Furnished Status", ["Furnished", "Semi-furnished", "Unfurnished"])
    floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=5)
    total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=20)

    st.markdown("---")

    # ── Neighbourhood ──
    st.markdown("### 🏫 Neighbourhood")
    nearby_schools = st.slider("Nearby Schools", 1, 10, 5)
    nearby_hospitals = st.slider("Nearby Hospitals", 1, 10, 3)
    transport = st.selectbox("Public Transport", ["High", "Medium", "Low"])

    st.markdown("---")

    # ── Amenities & Extras ──
    st.markdown("### ✨ Amenities & Extras")
    parking = st.selectbox("Parking Space", ["Yes", "No"])
    security = st.selectbox("Security", ["Yes", "No"])
    amenities_selected = st.multiselect("Amenities", ALL_AMENITIES, default=ALL_AMENITIES)
    facing = st.selectbox("Facing", ["North", "South", "East", "West"])
    owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
    availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])


# ─────────────────────────────────────────────
# Build feature vector
# ─────────────────────────────────────────────
def build_features():
    """Construct a single-row DataFrame that matches model.feature_names_in_."""
    current_year = 2026
    age = current_year - year_built

    # Median price per sqft from dataset as a reasonable default
    median_ppsf = df["Price_per_SqFt"].median()

    row = {feat: 0 for feat in feature_names}

    # Numeric features
    row["Size_in_SqFt"] = size_sqft
    row["Price_per_SqFt"] = median_ppsf
    row["Year_Built"] = year_built
    row["Age_of_Property"] = age
    row["Nearby_Hospitals"] = nearby_hospitals

    # One-hot: State
    key_state = f"State_{state}"
    if key_state in row:
        row[key_state] = 1

    # One-hot: City
    key_city = f"City_{city}"
    if key_city in row:
        row[key_city] = 1

    # One-hot: Locality
    key_loc = f"Locality_{locality}"
    if key_loc in row:
        row[key_loc] = 1

    # One-hot: Property Type
    key_pt = f"Property_Type_{property_type}"
    if key_pt in row:
        row[key_pt] = 1

    # One-hot: Public Transport
    key_tr = f"Public_Transport_Accessibility_{transport}"
    if key_tr in row:
        row[key_tr] = 1

    # One-hot: Security
    key_sec = f"Security_{security}"
    if key_sec in row:
        row[key_sec] = 1

    # One-hot: Availability
    key_av = f"Availability_Status_{availability}"
    if key_av in row:
        row[key_av] = 1

    # One-hot: Amenities (sorted combination string)
    amenities_str = ", ".join(sorted(amenities_selected))
    key_am = f"Amenities_{amenities_str}"
    if key_am in row:
        row[key_am] = 1
    else:
        # Try all permutations stored in model features
        for f in feature_names:
            if f.startswith("Amenities_"):
                stored_set = set(a.strip() for a in f.replace("Amenities_", "").split(","))
                if stored_set == set(amenities_selected):
                    row[f] = 1
                    break

    return pd.DataFrame([row], columns=feature_names)


# ─────────────────────────────────────────────
# Main content area
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🏠 India Housing Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Powered by a Random Forest model trained on 250 000+ real listings</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── Summary cards ──
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

# ── Predict button ──
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

    # ── Feature Importance ──
    st.markdown('<div class="section-title">📊 Top Feature Importances</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(15)

    fig = px.bar(
        feat_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#6366f1", "#a78bfa", "#c084fc", "#60a5fa", "#34d399"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="rgba(255,255,255,0.75)",
        font_family="Inter",
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        margin=dict(l=0, r=20, t=10, b=10),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Input summary table ──
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
            state, city, locality, property_type, bhk,
            f"{size_sqft:,}", year_built, furnished_status, floor_no,
            total_floors, nearby_schools, nearby_hospitals,
            transport, parking, security, ", ".join(amenities_selected),
            facing, owner_type, availability,
        ],
    }
    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True,
    )

# ─────────────────────────────────────────────
# Dataset exploration
# ─────────────────────────────────────────────
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
            color_continuous_scale=["#6366f1", "#a78bfa", "#60a5fa", "#34d399"],
            labels={"Price_in_Lakhs": "Avg Price (₹ Lakhs)"},
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="rgba(255,255,255,0.75)",
            font_family="Inter",
            coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=10),
            height=600,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Footer ──
st.markdown("""
<div class="footer-text">
    Built with ❤️ using Streamlit &amp; Scikit-learn · India Housing Price Predictor
</div>
""", unsafe_allow_html=True)
