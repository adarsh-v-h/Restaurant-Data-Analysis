# **Cuisine Classification (Final Report Section)**
---
## **Overview**
In this task, I developed a multi-class cuisine classifier to predict the primary cuisine of a restaurant using the structured features available in the processed dataset. <br>
This problem is challenging due to extreme class imbalance and limited discriminative features, but it serves as a valuable exercise in supervised learning and model evaluation.

---

## **Objective**
Build a machine learning model that classifies each restaurant into one of the cuisine groups based on features such as:
- Country Code
- Average Cost for Two
- Price Range
- Votes
- Has Table Booking
- Has Online Delivery
- City Frequency Encoding
Cuisines are represented using previously engineered one-hot groups (e.g., Cuisine_Grouped_North Indian, Cuisine_Grouped_Chinese, etc.).

## **Dataset Preparation**
**Target (y): All Cuisine Classes**
Each restaurant has exactly one active cuisine flag among ~40–50 one-hot columns.<br>
Using:
```python
y = df[cuisine_cols].idxmax(axis=1).str.replace("Cuisine_Grouped_", "")
```
I extracted the cuisine label for every row.<br>
**Features (X):**
X was created by removing:
- All cuisine one-hot columns (to prevent leaking answers)
- Aggregate rating (to avoid using model-generated signals)
The final feature set had:
```java 
X shape = (7403 rows, 7 features)
```
**Train/Test Split**
A stratified split was used to preserve class distribution:
```yaml
Train size: 5922
Test size : 1481
```

---
## **Model Used**
I trained a RandomForestClassifier as a baseline:
- 200 estimators
- balanced across cores
- default depth/leaf parameters
Random Forest was chosen because:
- it handles mixed numeric features well
- robust to noise
- fast to train
- provides feature importances
- strong baseline before more advanced models

---
## **Results**
```makefile
Overall Metrics (Baseline)
Accuracy: 0.2444  (24.4%)
Precision: 0.2263
Recall:    0.2444
F1:        0.2333 (weighted)
```
The model predicted correctly roughly 1 out of 4 times. <br>
Although low, this is expected given the nature of the problem.

---

## **Per-Class Analysis**
The classification report highlighted:
- North Indian (largest class) had the highest recall (~50%)
- Many small classes (BBQ, Goan, Bengali, Tibetan, Brazilian…) had 0% recall
- Some cuisines with moderate size like Café, Fast Food, Chinese had ~10–20% F1
- Extremely rare cuisines with <10 samples were never predicted
This is a textbook example of class imbalance + weak feature discrimination.

---

## **Confusion Matrix Observations**
The confusion matrix (saved as visuals/cuisine_confusion_matrix.png) showed:
- Strong diagonal only for large classes like North Indian
- Consistent confusion between:
-- Asian ↔ Chinese
-- Bakery ↔ Café
-- Mithai ↔ Bakery
-- Fast Food ↔ Continental
- Rare cuisines were always classified into larger, visually similar categories
- Many classes received no predictions at all due to insufficient training samples

---

## **Why the Accuracy is Low**
1. **Extreme class imbalance**
Some cuisines had:
- 400+ samples (North Indian)
- 3–5 samples (many minority cuisines)
Random Forest cannot learn predictable patterns from classes with <20 examples.

---

2. **Weak input features for cuisine prediction**
The model only had numeric / structural features:
- cost
- delivery flags
- booking flags
- country code
- votes
- city frequency
These features do not uniquely identify cuisine.<br>
Cake shops, cafés, sweet shops, burger joints, and fast food outlets all share similar patterns.<br>

---

3. **One-hot cuisine groups were removed from X**
Since we predicted cuisine, the cuisine columns cannot be used as features.<br>
This leaves very little information to classify highly diverse cuisines accurately.

---

## **What Could Improve the Model**
▶ 1. **Collapse rare classes into “Other”** <br>
Group all cuisines with <100 samples into a single Other class. <br>
This would immediately increase accuracy and stability. <br>

▶ 2. **Use text-based features** <br>
The original dataset contains fields like:
- Cuisines (string)
- Restaurant Name
- Locality
Using TF-IDF or sentence embeddings would dramatically improve discriminative power.<br>

▶ 3. **Use class balancing**
Options:
- Class_weight="balanced"
- RandomOverSampler
- SMOTE

▶ 4. **Consider a hierarchy**
Predict:
- broad cuisine family first
- then sub-cuisines
Parents: Indian / Asian / European / Fast Food / Cafe <br>
Children: Mughlai, Biryani, Tibetan, French, Italian, etc. <br>

▶ 5. **Consider stronger models**
- LightGBM
- XGBoost
- Neural embeddings

---

## **Conclusion**
This task demonstrates the challenges of multi-class classification on imbalanced, low-signal features. <br>
While the model's accuracy is modest, it provides a realistic baseline and valuable diagnostic insights into:
- dataset limitations
- representation issues
- class imbalance
- model behavior
The output models, confusion matrix, and feature importances are stored for review.