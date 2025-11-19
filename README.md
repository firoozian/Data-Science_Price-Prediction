# 🏠 Airbnb Price Prediction — EDA + XGBoost Model

This project builds an end-to-end pipeline to predict Airbnb listing prices across U.S. cities. It includes full exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## 📊 Exploratory Data Analysis (EDA)
• Price Distribution — histogram of cleaned listing prices  
• Price by Room Type — boxplot comparing room-type price differences  
• City-Level Median Prices — median price per city  
  **Insight:** median prices across cities fall between **$95 and $200**  
• Geospatial Visualization — U.S. map with:  
  - bubble size = listing density  
  - bubble color = average price  
  - labeled cities  

## ⚙️ Feature Engineering
- Log-transform applied to target (`price_log`)  
- StandardScaler for numeric features  
- One-hot encoding for categorical variables (`room_type`, `city`)  
- Removed invalid or extreme prices ($10–$5000)  
- Merged low-frequency cities into “Other”  

## 🤖 Modeling (XGBoost)
- XGBoost Regressor with tuned hyperparameters  
- Training on log-transformed target  
- 80/20 train–test split  
- Parallelized model training  

## 📏 Evaluation
- RMSE ≈ 234  
- MAE ≈ 87  
- R² ≈ 0.33  
Visual diagnostics include:  
- Actual vs Predicted scatter plot  
- Residual distribution histogram  

## 💾 Output
Final predictions saved to `Airbnb_XGB_Baseline_Results.csv` containing:  
ActualPrice, PredictedPrice

## 🧠 Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Cartopy

## 👤 Author
Sina Firoozian  
sina.firuzian@gmail.com  
