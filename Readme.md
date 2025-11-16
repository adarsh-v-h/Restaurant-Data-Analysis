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

## 🧪 How to Run the Project
```bash
# Create environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
