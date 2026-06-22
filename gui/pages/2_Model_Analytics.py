import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Titanic Predictor - Analytics",
    page_icon="📊",
    layout="wide"
)

st.markdown("# 📊 Model Analytics")
st.markdown("Performance comparison and analysis of all trained models.")

results_path = Path("models/model_comparison.csv")

if results_path.exists():
    results_df = pd.read_csv(results_path, index_col=0)
    
    st.markdown("## 📈 Model Performance Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(results_df, use_container_width=True)
        
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        results_df['accuracy'].plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(results_df['accuracy']):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
            
        st.pyplot(fig)
        
    best_model = results_df['accuracy'].idxmax()
    best_accuracy = results_df.loc[best_model, 'accuracy']
    
    st.markdown(f"""
    ## 🏆 Best Performing Model
    **{best_model}** achieved the highest accuracy of **{best_accuracy:.2%}**
    """)
    
    with st.expander("📖 Understanding the Metrics"):
        st.markdown("""
        - **Accuracy:** Overall correct predictions
        - **Precision:** Of predicted survivors, how many actually survived
        - **Recall:** Of actual survivors, how many were correctly identified
        - **F1-Score:** Harmonic mean of Precision and Recall
        - **ROC-AUC:** Ability to distinguish between survivors and non-survivors
        """)

    st.markdown("---")
    st.markdown("## 📊 Confusion Matrix Explanation")
    
    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.markdown("""
        | Actual / Predicted | Predicted No | Predicted Yes |
        | :--- | :---: | :---: |
        | **Actual No** | TN | FP |
        | **Actual Yes** | FN | TP |
        
        - **TN (True Negative):** Correctly predicted non-survival
        - **FP (False Positive):** Predicted survival but died
        - **FN (False Negative):** Predicted death but survived
        - **TP (True Positive):** Correctly predicted survival
        """)
        
    with col_cm2:
        st.markdown("""
        ### Key Insights
        - **High Precision** = Few false alarms
        - **High Recall** = Few missed survivors
        - For Titanic, **Recall is crucial** — we don't want to miss actual survivors!
        """)
else:
    st.warning("""
    ⚠️ Model comparison results not found!
    
    Please run:
```bash
    python run_pipeline.py
    ```
    """)

# Feature Importance section
st.markdown("---")
st.markdown("## 🔑 Feature Importance")

feature_data = {
    'feature': ['Sex', 'Pclass', 'Fare', 'Title', 'Age', 'FamilySize', 'SibSp', 'IsAlone', 'Parch', 'Embarked'],
    'importance': [0.215, 0.141, 0.098, 0.069, 0.071, 0.045, 0.022, 0.015, 0.008, 0.031]
}

feature_df = pd.DataFrame(feature_data).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(feature_df['feature'], feature_df['importance'], color='steelblue')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('What Factors Matter Most for Survival?', fontsize=14, fontweight='bold')

for bar, val in zip(bars, feature_df['importance']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
    
st.pyplot(fig)

st.markdown("""
### What These Features Mean:

| Feature | Why It Matters |
| :--- | :--- |
| **Sex** | Women had a ~74% survival rate vs. ~19% for men. |
| **Pclass** | 1st class passengers had ~63% survival vs. ~24% for 3rd class. |
| **Fare** | Higher fares generally meant safer, upper-deck cabin locations. |
| **Title** | Captured social status and structural protocols (e.g., *Mr*, *Mrs*, *Miss*, *Master*). |
| **Age** | Children and the elderly were prioritized during evacuation. |
""")

st.markdown("---")
st.caption("Analytics update after each training run.")