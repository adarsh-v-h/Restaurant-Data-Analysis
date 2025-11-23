# Task 1 — Predict Restaurant Ratings (src/task1)

This folder wraps Task 1 into runnable Python scripts. It uses saved modeling artifacts (models/, visuals/) created via the notebooks.

## How to run
From project root:

```bash
# quick evaluation of saved model on test split:
python src/task1/main.py --mode quick_eval

# predict a single index (e.g. 23):
python src/task1/main.py --mode single_predict --idx 23
```
---

## **Local SHAP Interpretation (Index 23)**:
For this restaurant, the model predicted its rating by heavily weighting the high number of votes and the popularity of its city (City_Freq). Its cuisine category had a mild positive effect, while moderate pricing nudged the prediction slightly downward. Overall, the SHAP breakdown shows the model’s logic is consistent with global feature patterns.