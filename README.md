
# AI Maternal Risk Predictor 🤰

An **AI-powered healthcare web application** that predicts the **risk level of maternal mortality during pregnancy** using machine learning and explainable AI. The system helps healthcare professionals identify high-risk pregnancies early and take preventive medical action.

# Project Overview

Maternal mortality is a major global health issue. Early detection of high-risk pregnancies can significantly reduce complications and improve maternal care.

This project uses **machine learning and clinical parameters** to predict the **risk level of pregnancy** and provide **explainable insights** to support medical decision-making.

# Key Features

• AI-based maternal risk prediction
• Explainable AI using **SHAP (Shapley values)**
• **Model confidence visualization** using gauge charts
• **Auto-generated patient ID system**
• **Returning patient tracking**
• **Patient history management**
• **Admin dashboard with analytics**
• **Patient search by Patient ID**
• **Risk trend analysis for each patient**
• **PDF medical report generation**

# Technology Stack

**Frontend / Web App**

* Streamlit

**Machine Learning**

* Scikit-learn
* SHAP (Explainable AI)

**Data Processing**

* Pandas
* NumPy

**Visualization**

* Plotly
* Matplotlib

**Database**

* SQLite

**Reporting**

* ReportLab (PDF generation)

# How the System Works

1. User enters patient clinical parameters:

   * Age
   * Blood Pressure
   * Blood Sugar
   * Body Temperature
   * Heart Rate

2. The trained machine learning model predicts the **maternal risk level**.

3. The system displays:

   * Risk Score
   * Risk Category (Low / Moderate / High)
   * Model Confidence
   * Explainable AI insights (SHAP)

4. The prediction is stored in the database with a **unique patient ID**.

5. Doctors or admins can:

   * Search patients
   * View patient history
   * Monitor risk trends

# Project Structure

AI-maternal-risk-predictor
│
├── app.py                 # Main Streamlit application
├── maternal_risk_model.pkl # Trained ML model
├── train.ipynb            # Model training notebook
├── assets/                # Images and UI assets
├── pages/                 # Dashboard pages
├── data/                  # Dataset
├── README.md              # Project documentation
```


# Future Improvements

• Integration with hospital electronic health records
• Real-time monitoring with IoT health devices
• Mobile health application
• Deployment on cloud healthcare platforms

---

# Impact

This system demonstrates how **AI and data-driven healthcare tools** can assist medical professionals in identifying high-risk pregnancies early and improving maternal healthcare outcomes.

 

