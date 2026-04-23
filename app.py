import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Estate Insight | India",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0f1115;
    background-image: 
        radial-gradient(circle at 20% 30%, rgba(197, 163, 108, 0.03) 0%, transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(99, 102, 241, 0.03) 0%, transparent 40%);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #16191f !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

section[data-testid="stSidebar"] .stMarkdown h2, 
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Playfair Display', serif;
    color: #c5a36c;
    letter-spacing: 0.02em;
}

section[data-testid="stSidebar"] label {
    color: rgba(255, 255, 255, 0.7) !important;
    font-size: 0.85rem;
    font-weight: 500;
}

/* Input Fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #1c2128 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px;
    color: #fff !important;
}

/* Hero Section */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}

.hero-sub {
    font-size: 1.1rem;
    color: #8a8f98;
    font-weight: 400;
    margin-bottom: 40px;
}

/* Cards */
.glass-card {
    background: rgba(30, 35, 45, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
    transition: transform 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(197, 163, 108, 0.3);
}

/* Metrics */
.metric-card {
    background: #1a1e26;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #c5a36c;
    margin-top: 4px;
}

.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}

/* Result Section */
.price-result {
    background: linear-gradient(180deg, #1e232d 0%, #16191f 100%);
    border: 1px solid #c5a36c44;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    margin: 32px 0;
}

.price-amount {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    color: #ffffff;
    margin: 8px 0;
}

.price-label {
    font-size: 1rem;
    color: #c5a36c;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    font-weight: 600;
}

/* Section Titles */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: #ffffff;
    margin-top: 48px;
    margin-bottom: 24px;
}

.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, #c5a36c 0%, transparent 100%);
    margin-bottom: 32px;
    opacity: 0.4;
}

/* Button */
.stButton > button {
    background-color: #c5a36c !important;
    color: #0f1115 !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #d4af37 !important;
    transform: scale(1.01);
}

/* Chat/Advisor */
.chat-bubble {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    border-left: 2px solid #c5a36c;
}

.footer-text {
    text-align: center;
    font-size: 0.8rem;
    color: #4b5563;
    margin-top: 80px;
    padding-bottom: 40px;
}
</style>
""", unsafe_allow_html=True)


BASE = Path(__file__).parent

load_dotenv(BASE / ".env")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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
    st.markdown("## Selection")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Location")
    state = st.selectbox("State", sorted(state_city_map.keys()), index=15)
    cities = state_city_map.get(state, [])
    city = st.selectbox("City", cities)
    localities = city_locality_map.get(city, [])
    locality = st.selectbox("Locality", localities)

    st.markdown("---")

    st.markdown("### Property Type")
    property_type = st.selectbox("Type", ["Apartment", "Independent House", "Villa"])
    bhk = st.slider("BHK", 1, 5, 2)
    size_sqft = st.slider("Size (SqFt)", 500, 5000, 1500, step=50)
    year_built = st.slider("Year Built", 1990, 2025, 2015)

    st.markdown("---")

    st.markdown("### Building Details")
    furnished_status = st.selectbox("Furnished Status", ["Furnished", "Semi-furnished", "Unfurnished"])
    floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=5)
    total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=20)

    st.markdown("---")

    st.markdown("### Neighbourhood")
    nearby_schools = st.slider("Nearby Schools", 1, 10, 5)
    nearby_hospitals = st.slider("Nearby Hospitals", 1, 10, 3)
    transport = st.selectbox("Public Transport", ["High", "Medium", "Low"])

    st.markdown("---")

    st.markdown("### Amenities & Extras")
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
st.markdown('<div class="hero-title">Estate Insight</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Sophisticated property valuation using advanced regression trained on 250,000+ premium listings</div>', unsafe_allow_html=True)
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
if st.button("Predict Price"):
    with st.spinner("Crunching numbers …"):
        input_df = build_features()
        prediction = model.predict(input_df)[0]
    st.session_state.prediction = prediction
    st.session_state.property_context = f"""
    Property Details:
    - Location: {locality}, {city}, {state}
    - Type: {property_type}, {bhk} BHK
    - Size: {size_sqft} sq.ft, Built in {year_built}
    - Floor: {floor_no} of {total_floors}
    - Furnished: {furnished_status}
    - Nearby Schools: {nearby_schools}, Hospitals: {nearby_hospitals}
    - Transport: {transport}, Parking: {parking}, Security: {security}
    - Amenities: {', '.join(amenities_selected) if amenities_selected else 'None'}
    - Facing: {facing}, Owner: {owner_type}, Status: {availability}
    - Predicted Price: ₹{prediction:,.2f} Lakhs
    """
    st.session_state.chat_history = []

if "prediction" in st.session_state:
    prediction = st.session_state.prediction

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
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Estate Concierge</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <p style="color: rgba(255,255,255,0.6); margin:0; font-size: 0.95rem;">
        Consult our expert advisor regarding investment potential, 
        market valuation, or neighborhood dynamics for this property.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        is_user = msg["role"] == "user"
        role_label = "You" if is_user else "Concierge"
        text_color = "#ffffff" if is_user else "#c5a36c"
        
        st.markdown(f"""
        <div class="chat-bubble">
            <span style="color: {text_color}; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">{role_label}</span>
            <p style="color: rgba(255,255,255,0.85); margin: 6px 0 0 0; font-size: 1rem; line-height: 1.5;">{msg["content"]}</p>
        </div>
        """, unsafe_allow_html=True)

    user_question = st.text_input(
        "Consultation query",
        placeholder="e.g. Is this a sound long-term investment?",
        key="ai_input"
    )

    if st.button("Consult Concierge", key="ask_ai"):
        if not GROQ_API_KEY:
            st.warning("Please set your GROQ_API_KEY in the .env file.")
        elif not user_question.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Consulting..."):
                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    system_prompt = f"""You are a highly sophisticated Indian real estate consultant.
                    A user has a property with these details and predicted price:
                    {st.session_state.property_context}
                    Answer their question in 3-4 sentences. Be specific, practical, and helpful.
                    Use Indian real estate context. Keep it conversational and clear."""

                    chat_response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_question}
                        ],
                        max_tokens=300
                    )
                    ai_reply = chat_response.choices[0].message.content
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                    st.rerun()

                except Exception as e:
                    st.error(f"AI Advisor error: {str(e)}")

    if st.session_state.chat_history:
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
st.markdown("")
st.markdown('<div class="section-title">📈 Dataset Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

with st.expander("Explore the training dataset", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Statistics", "Sample Data", "Price by City"])

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
    © 2026 Estate Insight • India Housing Price Predictor
</div>
""", unsafe_allow_html=True)
