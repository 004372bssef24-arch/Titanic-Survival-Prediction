import streamlit as st
import pandas as pd
import numpy as np  # Fixed: Added missing import to prevent error in correlation logic
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="Titanic Predictor - Data Explorer",
    page_icon="📈",
    layout="wide"
)

st.markdown("# 📈 Data Explorer")
st.markdown("Explore and visualize the Titanic dataset interactively.")

@st.cache_data
def load_data():
    data_path = Path("data/titanic.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

df = load_data()

if df is None:
    st.warning("""
    ⚠️ Dataset not found!
    
    Please download `titanic.csv` from Kaggle and place it in the `data/` folder.
    """)
    st.stop()

# Dataset info
st.markdown("## 📋 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Passengers", f"{len(df):,}")
with col2:
    st.metric("Features", len(df.columns))
with col3:
    survival_rate = df['Survived'].mean() * 100
    st.metric("Survival Rate", f"{survival_rate:.1f}%")
with col4:
    missing = df.isnull().sum().sum()
    st.metric("Missing Values", missing)

# Data preview
st.markdown("## 🔍 Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# Interactive filters
st.markdown("## 🎯 Interactive Filters")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    filter_class = st.multiselect("Filter by Class", [1, 2, 3], default=[1, 2, 3])
with col_f2:
    filter_sex = st.multiselect("Filter by Gender", ["male", "female"], default=["male", "female"])
with col_f3:
    filter_survived = st.multiselect("Filter by Survival", [0, 1], default=[0, 1], format_func=lambda x: "Survived" if x == 1 else "Did Not Survive")

filtered_df = df[
    (df['Pclass'].isin(filter_class)) &
    (df['Sex'].isin(filter_sex)) &
    (df['Survived'].isin(filter_survived))
]

st.markdown(f"**Showing {len(filtered_df)} passengers**")
st.dataframe(filtered_df, use_container_width=True)

# Visualizations
st.markdown("## 📊 Visualizations")
viz_type = st.selectbox(
    "Select Visualization",
    ["Survival by Class", "Survival by Gender", "Age Distribution", "Fare Distribution", "Correlation Heatmap"]
)

if viz_type == "Survival by Class":
    fig, ax = plt.subplots(figsize=(8, 6))
    survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
    bars = ax.bar(['1st Class', '2nd Class', '3rd Class'], survival_by_class.values, 
                  color=['#ffd43b', '#ff922b', '#ff6b6b'])
    ax.set_ylabel('Survival Rate (%)')
    ax.set_title('Survival Rate by Passenger Class', fontweight='bold')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, survival_by_class.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%', ha='center', fontweight='bold')
    st.pyplot(fig)

elif viz_type == "Survival by Gender":
    fig, ax = plt.subplots(figsize=(8, 6))
    survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100
    bars = ax.bar(['Female', 'Male'], survival_by_sex.values, color=['#51cf66', '#ff6b6b'])
    ax.set_ylabel('Survival Rate (%)')
    ax.set_title('Survival Rate by Gender', fontweight='bold')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, survival_by_sex.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%', ha='center', fontweight='bold')
    st.pyplot(fig)

elif viz_type == "Age Distribution":
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist([df[df['Survived'] == 0]['Age'].dropna(), df[df['Survived'] == 1]['Age'].dropna()],
            bins=25, label=['Did Not Survive', 'Survived'], alpha=0.7, color=['#ff6b6b', '#51cf66'])
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of Passengers')
    ax.set_title('Age Distribution by Survival Status', fontweight='bold')
    ax.legend()
    st.pyplot(fig)

elif viz_type == "Fare Distribution":
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['Fare'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Fare ($)')
    ax.set_ylabel('Number of Passengers')
    ax.set_title('Fare Distribution', fontweight='bold')
    ax.axvline(df['Fare'].median(), color='red', linestyle='--', label=f"Median: ${df['Fare'].median():.2f}")
    ax.legend()
    st.pyplot(fig)

elif viz_type == "Correlation Heatmap":
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontweight='bold')
    st.pyplot(fig)

# Statistics
st.markdown("## 📊 Summary Statistics")
stat_type = st.radio("Select Statistics", ["Numerical Features", "Categorical Features"], horizontal=True)

if stat_type == "Numerical Features":
    numeric_cols = df.select_dtypes(include=['number']).columns
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
else:
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        st.markdown(f"**{col}**")
        st.dataframe(df[col].value_counts().head(10), use_container_width=True)

st.markdown("---")
st.caption("Data source: Kaggle Titanic Dataset")