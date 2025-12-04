# **Exploratory Data Analysis (EDA) & Preprocessing**
---
## **Overview**
Task 1 focused on understanding the raw restaurant dataset and preparing it for machine learning. <br>
This stage is critical — most ML failures happen before modeling due to bad, missing, or unstructured data. <br>
The goal was to:
- Explore patterns, anomalies, and distributions
- Clean and standardize the dataset
- Engineer meaningful features
- Output a final processed dataset ready for modeling
This created the foundation for Tasks 2 and 3.

---

## **Dataset Understanding**
The dataset contained 21 original columns, including:
- Restaurant Name, City, Address, Locality
- Latitude & Longitude
- Cost for Two
- Has Online Delivery / Has Table Booking
- Primary cuisine string (comma-separated)
- Rating, Rating Text, Votes
- Country Code and Currency

Initial checks showed:
- **No missing values** in critical numeric columns
- Only **9 missing values** in Cuisines
- **No duplicate rows**
- Significant **categorical diversity**(50+ cuisine types, 100+ cities, 14 country codes)

---

## **Key Visual Insights**
### ⭐ 1. **Rating Distribution**
- Most ratings fall between 3.0 – 4.0
- Very few restaurants have extreme ratings (low outliers rare)
- KDE curve revealed a tight central tendency around 3.4

Interpretation:
Customer ratings on aggregator platforms tend to compress toward the middle — restaurants rarely get very low scores.

---

### ⭐ 2. **Cost Distribution**
- Highly skewed (long right tail)
- Most restaurants fall into the inexpensive to moderately priced range
- Log-scale histogram revealed clearer structure

Interpretation:
Restaurant pricing is not normally distributed — expensive places are rare, cheap places dominate.

---

### ⭐ 3. **Votes vs Ratings**
- Strong positive trend: higher votes → higher aggregate ratings
- Log-scale version confirmed stability at different magnitudes
- Votes become a major predictive feature (confirmed later via RandomForest)

Interpretation:
Rating credibility increases with vote count.

---

### ⭐ 4. **Top Cuisines & Top Cities**
- Top cuisines:
North Indian, Chinese, Fast Food, Bakery, Cafe
- Top cities:
New Delhi dominates heavily; then Gurgaon, Noida, Faridabad

Interpretation:
The dataset is heavily India-centric with a strong Delhi-NCR bias.

---

### ⭐ 5. **Correlation Heatmap**
- Votes, Price Range, and Cost-for-Two correlated moderately
- No multicollinearity issues
- Aggregate rating weakly correlated with most numeric features

Interpretation:
Predicting rating is non-trivial — requires combining multiple weak signals.

---

## Preprocessing & Feature Engineering
### ✔ **Cleaned Cuisines**
- Split multi-cuisine strings using .str.split(",")
- Extracted primary cuisine
- Grouped rare cuisines (<10) into “Other”
(Then later used one-hot grouped cuisines in modeling)

---

### ✔ **Encoded Categorical Variables**
- Converted Country Code into integer category codes
- Created City Frequency Encoding (how often each city appears)
- One-hot encoded major cuisine groups
- Removed irrelevant columns (Name, Address, Locality, Rating Color/Text)

---

### ✔ **Handling Flags**
Converted “Yes/No” flags into binary 1/0:
- Has Table Booking
- Has Online Delivery

---

### ✔ **Final Processed Dataset**
Shape:
```java
7403 rows × 53 columns
```
Saved as:
```bash
data/processed/model_data.csv
```
This is the dataset used for all modeling tasks.

---

## **Conclusion**
Task 1 established the entire foundation of the project:
- Clean dataset
- High-quality visual insights
- Well-designed features
- Structured preprocessing
- No missing values
- No data leakage
- Ready-to-use matrices for ML
