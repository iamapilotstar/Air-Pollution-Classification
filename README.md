# 🌫️ Beijing Air Pollution Level Classifier

> **Note**: This app is hosted on a free-tier Streamlit service that may go to sleep after periods of inactivity.  
If you encounter a loading delay, please be patient – the app is just waking up! It will be fully responsive after the initial load.

---

## 📋 Overview

This interactive application classifies Beijing air pollution levels based on meteorological conditions and pollutant concentrations.  
Leveraging machine learning (LightGBM), the app provides:

- Real-time pollution predictions  
- Comprehensive data visualizations  
- Analytical insights into the factors affecting air quality across Beijing  

---

## ⚙️ Installation


# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run weather_app.py



🌟 Features
Interactive Pollution Predictor: Input environmental parameters and receive immediate pollution level classification

Comprehensive EDA: Explore pollution patterns across regions, seasons, and area types

Model Performance Analysis: Compare various machine learning models with detailed metrics

Feature Importance Visualization: Understand the key factors driving pollution classification

SHAP Analysis: Gain insights into how the model interprets different environmental variables

Correlation Heatmap: Examine relationships between pollutants and meteorological conditions

🧪 Model Selection & Performance
After evaluating multiple classification algorithms, LightGBM was selected for its superior performance.

📊 Key Findings
Geographic Disparities: Urban areas experience ~25% higher pollution than rural ones

Seasonal Patterns: Winter pollution levels are ~40% higher than summer, especially in urban zones

Pollution Distribution:

High: 37.2%

Low: 37.7%

Moderate: 25.2%

Inverse Seasonal Trends:

SO₂, NO₂, and CO peak in winter

Ozone (O₃) peaks in summer

Influential Predictors:

PM10 concentration

Temp–dew point difference

SO₂ levels

🧠 Technical Approach
Data Preprocessing:

Log transformation for skewed variables

Label encoding for wind directions and stations

Feature Engineering:

Temperature-dew point difference

CO/NO₂ ratio

Inverse wind speed

Rain flag & night-time indicators

Model Training:

Ensemble models (LightGBM, CatBoost, etc.)

Hyperparameter tuning with GridSearchCV

Validation Strategy:

Stratified 5-fold cross-validation

Deployment:

Streamlit app for web interface

Plotly for interactive visualizations

🔮 Applications & Impact
🏥 Public Health Planning: Issue targeted advisories

🏙️ Urban Planning: Inform zoning and environmental design

📉 Policy Evaluation: Monitor long-term impact of regulations

🚶 Individual Awareness: Guide personal exposure decisions

📦 Requirements
Python 3.8+

Streamlit

NumPy

Pandas

Scikit-learn

LightGBM

Plotly
