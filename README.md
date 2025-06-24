# 🌫️ Beijing Air Pollution Level Classifier

> **Note**: This app is hosted on a free-tier Streamlit service that may go to sleep after periods of inactivity.  
If you encounter a loading delay, please be patient – the app is just waking up! It will be fully responsive after the initial load.

---

## 📋 Overview

Live Demo: https://airpollutionclassification.streamlit.app/

This interactive application classifies Beijing air pollution levels based on meteorological conditions and pollutant concentrations.  
Leveraging machine learning (LightGBM), the app provides:

- Real-time pollution predictions  
- Comprehensive data visualizations  
- Analytical insights into the factors affecting air quality across Beijing  

---

## ⚙️ Installation

pip install -r requirements.txt

# Run the application
streamlit run weather_app.py

----------------------------------------------------------------------------------------------------------------

🌟 Features
Interactive Pollution Predictor
- Input environmental parameters and receive immediate pollution level classification

Comprehensive EDA
- Explore pollution patterns across regions, seasons, and area types

Model Performance Analysis
- Compare multiple machine learning models with detailed evaluation metrics

Feature Importance Visualization
- Understand key factors driving pollution classification

SHAP Analysis
- Visualize how the model interprets different environmental variables

Correlation Heatmap
- Examine relationships between pollutants and meteorological factors

🧪 Model Selection & Performance
- Evaluated multiple classification algorithms (LightGBM, CatBoost, Random Forest, Logistic Regression)

- LightGBM was chosen for its high accuracy, efficiency, and interpretability

📊 Key Findings
Geographic Disparities
- Urban areas experience ~25% higher pollution than rural regions

Seasonal Patterns
- Winter months show ~40% higher pollution levels compared to summer, especially in urban zones

Pollution Distribution

- High: 37.2%

- Low: 37.7%

- Moderate: 25.2%

Inverse Seasonal Trends

- Primary pollutants (SO₂, NO₂, CO) peak in winter

- Ozone (O₃) peaks in summer

Most Influential Predictors

- PM10 concentration

- Temperature–dew point difference

- SO₂ levels

🧠 Technical Approach
🔧 Data Preprocessing
- Applied log transformation to reduce skewness in pollutant features

- Categorical encoding for wind directions and station locations

🧪 Feature Engineering
- Created new features including:

- Temperature–dew point difference

- CO/NO₂ ratio

- Inverse wind speed

- Night-time indicator

- Rain flag

🧠 Model Training
- Trained multiple models using ensemble learning methods

- Performed hyperparameter tuning via GridSearchCV

✅ Validation Strategy
- Used Stratified 5-Fold Cross-Validation for robust model performance estimation

🚀 Deployment
- Built an interactive Streamlit web application

Visualized data and model insights using Plotly charts

🔮 Applications & Impact
🏥 Public Health Planning
- Enables targeted health advisories based on predicted pollution levels

🏙️ Urban Planning
- Informs city zoning and infrastructure development based on pollution trends

📉 Policy Evaluation
- Assesses effectiveness of pollution control strategies over time

🚶 Individual Awareness
- Helps users make informed decisions about outdoor exposure

📦 Requirements
- Python 3.8+

- Streamlit

- NumPy

- Pandas

- Scikit-learn

- LightGBM
