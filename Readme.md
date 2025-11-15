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

## 📂 Project Structure
Cognifyz-ML-Internship/
│── data/
│ ├── raw/ ← original dataset
│ └── cleaned/ ← cleaned_dataset.csv (ready for ML)
│
│── notebooks/ ← EDA + modeling notebooks
│── src/ ← preprocessing + utilities + model scripts
│── models/ ← saved trained models
│── visuals/ ← plots and charts
│── reports/ ← task-wise PDF reports
│── README.md ← project documentation


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

### **Completed Till Now**
- [x] Load raw dataset  
- [x] Inspect dtypes, structure, missing values  
- [x] Remove irrelevant columns  
- [x] Extract *Primary Cuisine*  
- [x] Convert Yes/No → 1/0  
- [x] Save cleaned dataset  


## 🧪 How to Run the Project

```bash
# Create environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
