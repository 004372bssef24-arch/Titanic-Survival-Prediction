import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.prediction import Predictor

st.set_page_config(
    page_title="Titanic Predictor - Prediction",
    page_icon="🚢",
    layout="wide"
)

st.markdown("# 🔮 Make a Prediction")
st.markdown("Enter passenger details below to predict survival chances.")

@st.cache_resource
def get_predictor():
    try:
        return Predictor()
    except Exception as e:
        return None

predictor = get_predictor()

if predictor is None:
    st.error("""
    ❌ Models not loaded!
    
    Please run first:
```bash
    python run_pipeline.py
    ```
    """)
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Personal Information")
    name = st.text_input("Passenger Name", placeholder="Enter name (optional)")
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        format_func=lambda x: {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower)"}[x]
    )
    sex = st.radio("Gender", ["female", "male"], horizontal=True)
    age = st.number_input("Age (years)", min_value=0, max_value=100, value=30, step=1)

with col2:
    st.markdown("### 🎫 Travel Details")
    embarked = st.selectbox(
        "Embarkation Port",
        ["C", "Q", "S"],
        format_func=lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x]
    )
    fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=5.0)
    
    st.markdown("### 👨‍👩‍👧 Family Aboard")
    sibsp = st.number_input("Siblings/Spouse Aboard", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=6, value=0)

st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_btn = st.button("🔮 Predict Survival", type="primary", use_container_width=True)

if predict_btn:
    passenger_data = {
        'Name': name if name else "Passenger",
        'Pclass': pclass,
        'Sex': sex,
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
                st.success(f"### ✅ {result['prediction_text']}")
                st.metric("Survival Probability", f"{result['survival_probability']:.1%}")
                st.metric("Confidence", f"{result['confidence']:.1f}%")
                st.info(f"Risk Level: {result['risk_level']}")
            else:
                st.error(f"### ❌ {result['prediction_text']}")
                st.metric("Survival Probability", f"{result['survival_probability']:.1%}")
                st.metric("Confidence", f"{result['confidence']:.1f}%")
                st.warning(f"Risk Level: {result['risk_level']}")
                
            st.progress(result['survival_probability'])
            
            with st.expander("📖 View Detailed Explanation", expanded=True):
                st.markdown("#### Why this prediction?")
                st.write(result.get('explanation', 'No explanation available.'))
                
                st.markdown("#### Passenger Summary")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    class_suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(pclass, "Unknown")
                    st.metric("Class", class_suffix)
                    st.metric("Gender", sex.capitalize())
                with col_b:
                    st.metric("Age", f"{age} years")
                    st.metric("Fare", f"${fare:.2f}")
                with col_c:
                    family_size = sibsp + parch + 1
                    st.metric("Family Size", family_size)
                    st.metric("Embarked", embarked)
                    
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("⚠️ This is an educational project based on historical data.")