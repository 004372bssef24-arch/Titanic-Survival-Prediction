"""
Titanic Survival Prediction - Main GUI Application
Run with: streamlit run gui/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.prediction import Predictor

# Page configuration - MUST be first
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .survived {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
    }
    .not-survived {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🚢 Titanic Survival Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Will you survive? Enter passenger details below to find out.</div>', unsafe_allow_html=True)

# Load predictor
@st.cache_resource
def load_predictor():
    try:
        return Predictor()
    except Exception as e:
        return None

predictor = load_predictor()

if predictor is None:
    st.error("""
    ❌ Models not loaded! Please run first:
    ```bash
    python run_pipeline.py
    ```
    """)
    st.stop()

# Two column layout
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.markdown("### 📋 Passenger Information")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Passenger Name", placeholder="Enter name (optional)")

        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "🌟 1st Class (Upper)",
                2: "📘 2nd Class (Middle)",
                3: "⚓ 3rd Class (Lower)"
            }[x],
            help="Higher class had better access to lifeboats"
        )

        sex = st.radio("Gender", ["👩 Female", "👨 Male"], horizontal=True)
        sex_value = "female" if "Female" in sex else "male"
        # Ensure consistent format for predictor
        sex_value = sex_value.lower()

        age = st.number_input("Age (years)", min_value=0, max_value=100, value=30, step=1)

    with col2:
        embarked = st.selectbox(
            "Embarkation Port",
            options=["C", "Q", "S"],
            format_func=lambda x: {
                "C": "🇫🇷 Cherbourg (C)",
                "Q": "🇮🇪 Queenstown (Q)",
                "S": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Southampton (S)"
            }[x],
            help="Port where passenger boarded the Titanic"
        )

        fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=5.0)

        st.markdown("### 👨‍👩‍👧 Family Aboard")
        sibsp = st.number_input("Siblings/Spouse", min_value=0, max_value=8, value=0,
                                help="Number of siblings or spouses aboard")
        parch = st.number_input("Parents/Children", min_value=0, max_value=6, value=0,
                                help="Number of parents or children aboard")

with col_right:
    st.markdown("### 📊 Historical Statistics")
    st.markdown("---")

    st.markdown("""
    <div class="stat-box">
        <span style="font-size: 2rem;">38.4%</span><br>
        Overall Survival Rate
    </div>
    """, unsafe_allow_html=True)

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown("""
        <div class="info-box">
            <b>👩 Women</b><br>
            <span style="font-size: 1.5rem; color: #28a745;">74.0%</span>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown("""
        <div class="info-box">
            <b>👨 Men</b><br>
            <span style="font-size: 1.5rem; color: #dc3545;">18.9%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### By Class")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown('<div class="metric-card"><b>🌟 1st Class</b><br><span style="color:#28a745; font-size:1.3rem;">62.9%</span></div>', unsafe_allow_html=True)
    with col_c2:
        st.markdown('<div class="metric-card"><b>📘 2nd Class</b><br><span style="color:#ffc107; font-size:1.3rem;">47.3%</span></div>', unsafe_allow_html=True)
    with col_c3:
        st.markdown('<div class="metric-card"><b>⚓ 3rd Class</b><br><span style="color:#dc3545; font-size:1.3rem;">24.2%</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### By Age")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown('<div class="metric-card"><b>👶 Children (&lt;12)</b><br><span style="color:#28a745;">59.8%</span></div>', unsafe_allow_html=True)
    with col_a2:
        st.markdown('<div class="metric-card"><b>👴 Seniors (60+)</b><br><span style="color:#dc3545;">22.7%</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### By Port")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown('<div class="metric-card"><b>🇫🇷 C</b><br><span style="color:#28a745;">55.4%</span></div>', unsafe_allow_html=True)
    with col_p2:
        st.markdown('<div class="metric-card"><b>🇮🇪 Q</b><br><span style="color:#ffc107;">38.9%</span></div>', unsafe_allow_html=True)
    with col_p3:
        st.markdown('<div class="metric-card"><b>🇬🇧 S</b><br><span style="color:#dc3545;">33.6%</span></div>', unsafe_allow_html=True)

# Predict button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_btn = st.button("🔮 PREDICT SURVIVAL", type="primary", use_container_width=True)

# Handle prediction
if predict_btn:
    passenger_data = {
        'Name': name if name else "Passenger",
        'Pclass': pclass,
        'Sex': sex_value,
        'Age': age,
        'Embarked': embarked,
        'Fare': fare,
        'SibSp': sibsp,
        'Parch': parch
    }
    
    with st.spinner("Analyzing passenger data..."):
        try:
            result = predictor.predict_with_explanation(passenger_data)
            
            st.markdown("---")

            if result['survived']:
                st.markdown(f"""
                <div class="prediction-card survived">
                    <h1 style="color: #155724;">✅ {result['prediction_text']}</h1>
                    <p style="font-size: 1.2rem;"><strong>{passenger_data['Name']}</strong> would have survived the Titanic disaster!</p>
                    <div style="background: white; border-radius: 10px; padding: 0.5rem; margin: 0.5rem 0;">
                        <p style="font-size: 2rem; margin: 0;">{result['survival_probability']:.1%}</p>
                        <p>Survival Probability</p>
                    </div>
                    <p><strong>Confidence:</strong> {result['confidence']:.1f}%</p>
                    <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card not-survived">
                    <h1 style="color: #721c24;">❌ {result['prediction_text']}</h1>
                    <p style="font-size: 1.2rem;"><strong>{passenger_data['Name']}</strong> likely would not have survived</p>
                    <div style="background: white; border-radius: 10px; padding: 0.5rem; margin: 0.5rem 0;">
                        <p style="font-size: 2rem; margin: 0;">{result['survival_probability']:.1%}</p>
                        <p>Survival Probability</p>
                    </div>
                    <p><strong>Confidence:</strong> {result['confidence']:.1f}%</p>
                    <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                </div>
                """, unsafe_allow_html=True)

            # Probability gauge
            st.progress(result['survival_probability'])

            with st.expander("📖 View Detailed Explanation", expanded=True):
                st.markdown("#### Why this prediction?")
                st.write(result.get('explanation', 'No explanation available.'))

                st.markdown("#### Passenger Summary")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    class_label = {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[pclass]
                    st.metric("Class", class_label)
                    st.metric("Gender", "Female" if sex_value == "female" else "Male")
                with col_b:
                    st.metric("Age", f"{age} years")
                    st.metric("Fare", f"${fare:.2f}")
                with col_c:
                    family_size = sibsp + parch + 1
                    st.metric("Family Size", family_size)
                    port_label = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[embarked]
                    st.metric("Embarked", port_label)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Make sure you've trained the models first with: python run_pipeline.py")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.8rem;">
    ⚠️ This is an educational project based on historical data. Predictions are probabilistic.<br>
    📊 Data source: Kaggle Titanic Dataset | © 2024 Titanic Survival Prediction - Group 2, IIUI
</div>
""", unsafe_allow_html=True)