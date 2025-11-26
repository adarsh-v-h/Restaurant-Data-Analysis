# Content-based + Hybrid Recommender (src/task2)

## What this does
-Location: src/task2/ <br>
Task 2 implements a clean, production-style content-based recommender system using the feature vectors produced in Task 1.<br>
This system can:

- ✓ Predict the rating of a restaurant (from your tuned RandomForest model)
- ✓ Recommend the top-K most similar restaurants using cosine similarity
- ✓ Display human-readable metadata (name, city, cuisines) from the raw dataset
- ✓ Work entirely from your pre-saved joblib files (X_train/X_test/y_train/y_test + model + feature list)

Everything runs without reprocessing, retraining, or touching Jupyter notebooks

## How to run
From project root:

```bash
python src/task2/main.py
```
The program will ask you to enter an index, and you do so, will find out the top 5 restaurants similar to your choosen restaurant. It does it by building up a matrix using:
- Average Cost for two
- Price range
- Delivery flag
- Table booking flag
- Votes
- Country Code
- All cuisine one-hot flags
These features are scaled with StandardScaler to avoid magnitude domination.Cosine similarity is then computed on this processed vector space. And based on those similarities we find out **Top 5 restaurants similar to your choosen restaurant**