# 🌊 Urban Flood Prediction (Indonesia)

A machine learning pipeline to predict urban floods using daily climate data.

## 🧱 Pipeline Overview
1. **Data Cleaning** – Handles missing values per station  
2. **Feature Engineering** – Adds rainfall memory, humidity & temperature trends  
3. **Class Balancing** – SMOTE for rare flood events  
4. **Model Training** – XGBoost with precision-recall optimization  
5. **Visualization** – PR curve & flood risk distribution  

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/urban-flood-prediction.git
cd urban-flood-prediction
pip install -r requirements.txt
