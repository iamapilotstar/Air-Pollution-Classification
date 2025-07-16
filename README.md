# 🌍 BreatheSmart: Real-Time ML Forecasting for Air Pollution Trends

🔗 Live Demo:  https://airpollutionclassification.streamlit.app/

# 💡 The Problem

City dwellers and policymakers lack real-time, interpretable tools to monitor and act on rising pollution levels.
Delayed detection of hazardous air quality can lead to increased health risks, environmental damage, and missed opportunities for preventive intervention.

## 🔧 The Solution

I built a real-time machine learning application that classifies pollution levels into Low, Moderate, or High using Beijing’s weather and pollutant data.
It leverages LightGBM for high performance and SHAP-based interpretability to ensure transparency and actionable insights for both citizens and urban planners.

## 📌 Key Results

📊 Model Accuracy: 87.1%

- ROC-AUC: 96.6%

- Recall (High Pollution): 87.3% (optimized to minimize false negatives)

- Optimizing recall helps ensure high-risk pollution events aren't missed — critical for public safety.

## 🎯 Value Delivered

- Combines environmental science and machine learning to deliver a deployable early-warning tool

- Empowers both the public and local authorities to:

- Issue health advisories

- Optimize traffic flow

- Inform sustainable city policies

- Brings predictive power + explainability = trustworthy, transparent, and responsive environmental monitoring

## ⚙️ Installation

- pip install -r requirements.txt

# Run the application
streamlit run weather_app.py


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


## 📁 Folder Structure

```bash

Air-Pollution-Classification/
│
├── App and Analysis/
│   ├── weather_app.py
│   └── Air_Pollution_EDA_Analysis.ipynb
│
├── Models/
│   ├── lightgbm_model.pkl
│   └── scaler.pkl
│
├── Images/
│   ├── shap_summary.png
│   ├── feature_importance.png
│   ├── correlation_heatmap.png
│   └── pollution_distribution_chart.png
│
├── requirements.txt
└── README.md
