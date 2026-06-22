# 🚢 Titanic Survival Prediction System

## 📋 Project Overview
This is a complete **Machine Learning system** that predicts whether a passenger would have survived the Titanic disaster. The system uses historical passenger data to train multiple ML models and provides an interactive web application for making predictions.

## 🎯 Objectives
- Predict Titanic passenger survival using ML
- Build a complete EDA pipeline with visualizations
- Train and compare multiple ML models
- Deploy an interactive web application

## 🔧 Tools & Technologies
- **Language:** Python 3.10+
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Framework:** Streamlit
- **Model Serialization:** Joblib
- **Version Control:** Git & GitHub

## 📊 Key Features
- ✅ Exploratory Data Analysis with 15+ visualizations
- ✅ Feature Engineering (FamilySize, IsAlone, Title, etc.)
- ✅ 8 ML Models (Random Forest: 82.7% accuracy)
- ✅ Interactive Streamlit GUI
- ✅ Real-time survival predictions with explanations

## 📁 Project Structure
titanic-survival-prediction/
├── data/ # Dataset
├── src/ # Source code modules
├── gui/ # Streamlit web app
├── models/ # Saved models (.pkl)
├── notebooks/ # Jupyter notebooks
├── reports/ # Figures and outputs
├── requirements.txt # Dependencies
├── run_pipeline.py # Main execution script
└── README.md # Project documentation

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction.git
cd titanic-survival-prediction
2. Create Virtual Environment
bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
3. Install Dependencies
bash
pip install -r requirements.txt
4. Download Dataset
Place titanic.csv in the data/ folder (download from Kaggle)

5. Train the Models
bash
python run_pipeline.py
6. Launch the GUI
bash
streamlit run gui/app.py
📊 Model Performance
Model	Accuracy	ROC-AUC
Random Forest	82.7%	0.879
Gradient Boosting	82.1%	0.874
Logistic Regression	81.0%	0.863
👥 Team Members
M. Hasnain Baig – 4372/BSSE/F24 (Project Lead)

Iqbal Hassan Tariq – 4361/BSSE/F24

Habib Ullah – 4380/BSSE/F24

Nasrullah Malik – 4583/BSSE/F24

📚 Course Information
Course: Artificial Intelligence

Department: Software Engineering

University: International Islamic University, Islamabad (IIUI)
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
