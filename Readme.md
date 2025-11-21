# 🍽️ Cognifyz Machine Learning Internship – Restaurant Data Analysis

## 📍 Project Overview
This project analyzes restaurant data from multiple countries to uncover insights about:
- Cuisines
- City patterns
- Ratings
- Costs
- Customer votes
- Delivery & table booking patterns

The goal is to clean the dataset, understand patterns, build ML models, and finally present insights in a simple way that both technical and non-technical people can understand.

---

## 📘 Dataset Summary

- **Rows:** 9,551  
- **Columns (after cleaning):** 12  
- **Types of data:**  
  - Categorical (City, Country)  
  - Text (Cuisines)  
  - Numerical (Cost, Rating, Votes)  
  - Geo-coordinates (Lat/Long)

Key features include:
- Restaurant location  
- Primary cuisine  
- Average cost  
- Delivery & booking availability  
- Rating + Votes  

---

## 🔧 Tech Stack

- **Python** (Pandas, NumPy, Scikit-Learn)
- **JupyterLab** for EDA & experimentation    

---

<h3>📂 Project Structure</h3>

<pre>
├── data/
│   ├── raw/                # Original dataset
│   ├── cleaned/            # Cleaned CSV after preprocessing
│   └── processed/          # Final ML-ready dataset
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_visualizations.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_baseline_models.ipynb
├── visuals/                # All generated plots
├── models/
│   │── best_model_random_forest.joblib       # saved model
│   │── feature_list.joblib                   # required for deployment
│   └── rf_predictions.joblib                 # predictions + ground truth
├── reports/                # Final PDF reports (EDA + ML + Summary)
└── README.md
</pre>

---

## ✔️ Progress Checklist

### **Dataset Processing**
- [x] Load raw dataset  
- [x] Inspect dtypes, structure, missing values  
- [x] Remove irrelevant columns  
- [x] Extract *Primary Cuisine*  
- [x] Convert Yes/No → 1/0  
- [x] Save cleaned dataset  

---

### **What we understood from the dataset, EDA Visualizations**
- Most ratings concentrate between **2.8–4.2**, with **0.0** representing unrated restaurants.
- Cost distribution is extremely skewed, requiring log transformation to reveal true spending patterns.
- Votes correlate positively with both rating and price range—popular restaurants tend to be better rated and slightly more premium.
- The dataset is geographically skewed toward **Delhi NCR**, with international entries evenly distributed.
- Strong correlations exist between **Price Range ↔ Average Cost** and **Price Range ↔ Table Booking**, revealing consumer segmentation.
- Rating is not strongly influenced by cost, delivery availability, or table booking—indicating deeper factors like food quality or service.
- You can view the Plots in the /visuals directory.

---

### **Feature Engineering Summary**
- Removed unrated restaurants (rating = 0.0)
- Frequency encoded City to capture restaurant density
- Grouped rare cuisines (<10 occurrences) into "Other"
- One-hot encoded the final Cuisine_Grouped column
- Converted Country Code into categorical codes
- Dropped unnecessary text and geo-location columns (name, address, lat/long, etc.)
- Created df_model — a fully numeric, ML-ready dataset
- Performed an 80/20 train–test split

---

### **Baseline Models and Results**
1. **Linear Regression**
- R²: 0.43
- MAE: 0.33
- RMSE: 0.41
Conclusion: Simple models fail to capture the non-linear nature of restaurant ratings. Underfits the data.<br>

2. **Decision Tree**
- R²: 0.31
- MAE: 0.33
- RMSE: 0.46
Conclusion: Captures some relationships, but heavily overfits and lacks generalization.

3. **Random Forest (Best Baseline Model)**
- R²: 0.59
- MAE: 0.26
- RMSE: 0.35
Conclusion: 
- Learns complex, non-linear relationships
- Stable, balanced predictions
- Lowest error
- Behaves well across full rating range
This model is now saved in the /models directory for reuse.

---

## 🧪 How to Run the Project
```bash
# Create environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate for windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

---

## Highlights
- Dataset from 15+ countries
- 9,551 restaurants analyzed
- 50+ engineered features (including cuisines, pricing, frequency encoding)
- Rich visual analysis stored in /visuals
- Scalable feature engineering pipeline
- ML-ready dataset with 7 numeric + 45 one-hot features
- Random Forest has shown the best results when compared to other models, therefore it will be used as baseline model