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
## Project Structure

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
├── models/                 # Saved models, scalers, encoders, splits
├── reports/                # Final PDF reports (EDA + ML + Summary)
└── README.md

---
## ✔️ Progress Checklist

### **Dataset Processing**
- [x] Load raw dataset  
- [x] Inspect dtypes, structure, missing values  
- [x] Remove irrelevant columns  
- [x] Extract *Primary Cuisine*  
- [x] Convert Yes/No → 1/0  
- [x] Save cleaned dataset  

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

## 🧪 How to Run the Project
```bash
# Create environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate

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