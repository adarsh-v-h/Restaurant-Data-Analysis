# Task 2 — Content-based + Hybrid Recommender (src/task2)

## What this does
- Filters restaurants by cuisine and city.
- Optionally uses cuisine-vector cosine similarity to find items similar to a chosen restaurant.
- Uses your trained RandomForest model to predict ratings for candidate items and computes a hybrid score combining model rating + normalized votes.

## How to run
From project root:

```bash
python src/task2/main.py --cuisines "North Indian,Italian" --city "New Delhi" --k 10
```
To get recommendations similar to an item index (useful for "people also liked"):
```bash
python src/task2/main.py --use_similarity --item_idx 23 --k 10
```