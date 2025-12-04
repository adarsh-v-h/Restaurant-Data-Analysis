# **Restaurant Rating & Insights** 
Full ML Workflow → EDA → Feature Engineering → Modeling → Explainability → Recommender System → Classification.

---

## 📍 **Project Overview**
This project explores and models restaurant data across multiple countries. <br>
The objective is to build a complete ML pipeline that can: 
- Analyze restaurant patterns (EDA)
- Predict restaurant ratings using machine learning
- Find similar restaurants using content-based recommendation
- Classify restaurants by cuisine using supervised learning
- Explain the model’s decisions using SHAP
This file contains the full summary of the work done across all tasks.

---

## 📊 **Dataset Summary**
**Raw Rows**: 9,551 <br>
**Processed Rows**: ~7,400 (after removing invalid/unrated rows) <br>
**Final engineered features**: 53 <br>
The dataset includes:
- Restaurant Name, Location, Cuisine
- Average Cost for Two
- Delivery & Table Booking flags
- Country Code & Currency
- Rating + Votes
- Geo-coordinates
- Cuisines (multi-label string)
After preprocessing:
- All text columns removed/encoded
- Cuisine groups one-hot encoded
- City frequency encoded
- Country Code converted to category codes
- Flags converted to binary
- Final numeric ML-ready dataset saved at:
data/processed/model_data.csv

---

## 📂 **Project Structure**
<prev>
Cognifyz-ML-Internship/
│
├── data/
│   ├── raw/                 # original dataset (Dataset.csv)
│   └── processed/           # cleaned ML-ready dataset (model_data.csv)
│
├── models/                  # saved models, splits, predictions, feature lists
│
├── visuals/                 # EDA plots, SHAP plots, confusion matrix, PDPs
│
├── notebooks/               # full exploratory Jupyter workflows
│   ├── 01_visualizations.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_model_interpretation.ipynb
│
├── src/
│   ├── task1/               # (optional) model loading utilities
│   ├── task2/               # content-based recommender system
│   │   ├── main.py
│   │   ├── data_utils.py
│   │   ├── recommend_utils.py
│   │   └── meta_utils.py
│   ├── task3/               # cuisine classification
│   │   ├── main.py
│   │   ├── data_utils.py
│   │   ├── train_utils.py
│   │   └── eval_utils.py
│   └── ...
│
├── README.md                # you're reading it
└── requirements.txt
</prev>

---

## 🧪 **Task 1 — Full EDA & Preprocessing Summary**
### **Key Visual Insights**
- Ratings are tightly centered between 3.0 and 4.0 — customers rarely give extreme ratings.
- Restaurant costs are extremely skewed — requiring log transformation.
- Votes are the strongest indicator of rating — more votes = higher reliability.
- Top cities dominated by NCR region — dataset is India-heavy.
- Top cuisines:
North Indian, Chinese, Fast Food, Bakery, Cafe.
All plots are stored under /visuals:
- rating_distribution
- cost_distribution
- log cost distribution
- votes vs rating
- top cities
- top cuisines
- correlation heatmap

---
### **Preprocessing Steps**
Extracted Primary Cuisine from multi-label cuisine strings. Grouped rare cuisines (<10) into “Other”.<br>
Encoded categorical variables:
- One-hot for cuisine groups
- Category codes for Country Code
- Frequency encoding for City

Converted Yes/No → 1/0, removed irrelevant columns (name, address, geolocation, text flags), ensured zero missing values, split dataset into X_train, X_test, y_train, y_test and saved all processed matrices & feature lists.

---

### **Output of Task 1**
- Fully numeric ML-ready dataset
- 53 engineered features
- No missing values
- No leakage
- Highly structured pipeline
- Used in both Tasks 2 and 3

---

## 🤖 **Task 2 — Content-Based Recommendation System**
The recommender selects a restaurant (by index) and returns:
- Predicted rating using the tuned RandomForest model
- Top-5 most similar restaurants using cosine similarity
- Full metadata (name, city, cuisine) from the raw dataset

**How similarity is computed**
Instead of using all 53 features, only the meaningful ones are included:
- Average cost
- Price range
- Votes
- Booking & delivery flags
- All cuisine-group flags
- Country code
Then standardized (scaled), and cosine similarity is computed.

**Why similarity is < 1.0 now**
The earlier issue of "1.0 similarity for everything" was fixed:
- We now use correct feature subset
- We apply scaling
- We exclude City_Freq
- We remove features that distort vectors
Now similarity values appear realistic (0.4–0.75).

---

**Example Output**
```python-repl
>>> SELECTED RESTAURANT
Name: Ikreate
City: New Delhi
Cuisines: Bakery
Predicted Rating: 3.17

>>> SIMILAR RESTAURANTS
1. A Pizza House    (Similarity: 0.67)
2. Tpot             (Similarity: 0.66)
3. Pandit Dhaba     (Similarity: 0.65)
...
```
**Files Implemented**
- data_utils.py – loads joblib splits
- recommend_utils.py – cosine similarity engine
- meta_utils.py – retrieves original metadata
- main.py – user interface

---

## 🍽️ **Task 3 — Cuisine Classification (Multiclass ML)**
Goal: classify a restaurant’s primary cuisine from its numeric features. <br>
**Why this task is hard?**
- 50+ cuisine labels
- Many classes very small (2–5 examples)
- Heavy class imbalance
- Cuisine is a high-level concept not captured well by numeric features

**Model Used**
RandomForest Classifier (baseline)
**Results**
```yaml
Accuracy: ~24%
Weighted F1: ~23%
```
**Interpretation**
- Model captures large classes (North Indian, Cafe, Chinese)
- Fails on very small cuisine groups
- Numeric features alone don’t represent true cuisine characteristics

**Confusion Matrix & Feature Importances**
Saved in /visuals:
- cuisine_confusion_matrix.png
- cuisine_feature_importances.png
**Conclusion**
This is a meaningful but challenging task. <br>
Strong improvement would require:
- Text embeddings for cuisine strings
- Richer menu information
- NLP-based cuisine similarity modeling

---

## 🧠 **Model Interpretation (SHAP)**
We used SHAP to explain the RandomForest rating predictor. <br>
**SHAP Summary Plot**
Shows global feature importance:
- Votes is the strongest driver
- City frequency contributes heavily
- Price & cost form moderate influence
- Cuisines influence rating non-linearly

**Local Explanation (Waterfall & Force Plot)**
For an example restaurant:
- High votes → pushes rating upward
- High cost → small upward push
- Specific cuisine flags → small adjustments
Interactive HTML and PNG versions saved in /visuals.

---

## 📈 **Hyperparameter Tuning (RandomForest)**
Using RandomizedSearchCV:

**Best Parameters:**
```makefile
n_estimators: 700
max_depth: 20
min_samples_split: 10
max_features: sqrt
bootstrap: True
```

**Final Test Results:**
```makefile
R2: 0.626
MAE: 0.256
RMSE: 0.339
```
Significant improvement over baseline.

---

## 🔧 **How to Run**
**1. Install environment**
```bash
python -m venv test
.\test\Scripts\activate
pip install -r requirements.txt
```
**2. Run recommender (Task 2)**
```bash
python src/task2/main.py
```
**3. Run cuisine classifier (Task 3)**
```bash
python src/task3/main.py
```

## **Final Deliverables**
✔ Cleaned dataset (model_data.csv) <br>
✔ Complete EDA visual package <br>
✔ Engineered feature matrix (53 features) <br>
✔ RandomForest rating model (baseline + tuned) <br>
✔ SHAP explainability <br>
✔ Content-based Recommender System <br>
✔ Cuisine Classification Model <br>
✔ Full project code (src/) <br>
✔ Production-ready README (this file) <br>