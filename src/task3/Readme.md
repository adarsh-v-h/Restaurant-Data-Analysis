# Cuisine Classification (src/task3)

This module trains a multi-class classifier to predict a restaurant's primary cuisine using the processed dataset.

How to run:

```bash
python main.py
```
## Output saved:
- models/cuisine_rf.joblib — trained classifier
- models/cuisine_feature_list.joblib — feature order used for training
- models/cuisine_preds.joblib — saved test predictions
- visuals/cuisine_confusion_matrix.png — confusion matrix heatmap
- visuals/cuisine_feature_importances.png — top feature importances