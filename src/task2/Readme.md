# **Content-Based Restaurant Recommender**
## **Overview**
Task 2 implements a content-based recommendation engine using the features engineered in Task 1. <br>
This component identifies restaurants that are similar to a chosen restaurant based on:
- cost
- price range
- delivery flags
- booking flags
- votes
- cuisine group vectors
It also integrates the trained RandomForest rating model to predict the restaurant’s expected rating.

---

## **Objective**
When given a restaurant:
- Predict its likely rating
- Compute cosine similarity with all other restaurants
- Return the top-K similar restaurants
- Display detailed metadata from the raw dataset
- Ensure the mapping between processed features and raw tables is accurate
This is a full workflow similar to ML-powered recommendation systems used in real-world platforms.

---

## **How It Works**
### ⭐ 1. **Uses Pre-Saved ML Artifacts**
Loaded from models/:
- final_rf_tuned.joblib
- X_train.joblib / X_test.joblib
- Y_train.joblib / Y_test.joblib
- feature_list.joblib
This avoids recomputing anything and ensures correct feature order.

---

### ⭐ 2. **Accurate Metadata Mapping**
Because ML matrices lose original indexes, Task 2 reconstructs metadata by aligning:
```java
X_test → final_features → mapping → Dataset.csv
```
This guarantees that the restaurant displayed is the correct one.

---

### ⭐ 3. **Feature Processing for Similarity**
The recommender uses only the features meaningful for content similarity:
- Average Cost
- Price Range
- Has Table Booking
- Has Online Delivery
- Votes
- Country Code
- All Cuisine_Grouped_* flags
Numeric features are scaled; cuisine flags are left as-is. <br>
This prevents features with large numbers (like votes) from dominating the similarity.

---

### ⭐ 4. **Cosine Similarity for Recommendations**
Cosine similarity measures: How similar are two restaurants based on direction, not magnitude?
Perfect for comparing:
- cost levels
- cuisine profiles
- city frequency structures
- This returns the Top-5 most similar restaurants.

---

### ⭐ 5. **Predicted Rating Display**
Before showing recommendations, the tuned RandomForest regressor predicts the selected restaurant’s rating (typically with ~0.34 RMSE).

---

## **Example Output**
```yaml
>>> SELECTED RESTAURANT
Name: Ikreate
City: New Delhi
Cuisine: Bakery
Predicted Rating: 3.17

>>> SIMILAR RESTAURANTS

1. A Pizza House (New Delhi)
   Cuisine: Pizza, Fast Food
   Similarity: 0.67

2. Tpot (Gurgaon)
   Cuisine: Cafe
   Similarity: 0.66
```
Similarity values are now realistic (0.4–0.7 range), not 1.0, because the correct feature subset and scaling were used.

---

## **Files Implemented**
**data_utils.py** loads pre-saved joblib datasets. <br>
**recommend_utils.py** handles:
- vector construction
- scaling
- cosine computation
- prediction
- top-K selection

**meta_utils.py** Maps ML rows → original raw dataset for display. <br>
**main.py** is CLI entry point.

---

## **Why This Works Well**
Task 2 succeeds because:
- The features actually represent similarity
- Cosine similarity is ideal for sparse, high-dimensional vectors
- The model predicts ratings before showing similar restaurants
- Metadata mapping ensures clean outputs
- Output is human-friendly and accurate
This creates a recommendation module similar to Zomato/Swiggy’s “similar restaurants” feature.

---

## **Conclusion**
Task 2 delivers a complete and functional Content-Based Recommendation Engine with:
- Reliable similarity computation
- Meaningful comparisons
- Integrated ML rating predictions
- Accurate metadata retrieval
- Clean Python module structure
This forms the “intelligent suggestion” component of the project.