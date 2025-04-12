import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the CSV files
df_baseline = pd.read_csv("manual_annotations.csv")
df_pred = pd.read_csv("llm_output_testing.csv")

# Merge dataframes on Article ID to ensure we compare same articles
merged_df = pd.merge(df_baseline, 
                    df_pred, 
                    on='Article ID', 
                    suffixes=('_baseline', '_pred'))

# Extract matched labels
baseline_categories = merged_df['Category_baseline'].tolist()
pred_categories = merged_df['Category_pred'].tolist()

# Get unique labels
labels = sorted(list(set(baseline_categories)))
print("Category Labels:", labels)
print(f"Total matched articles: {len(baseline_categories)}")

# Compute confusion matrix for categories
cm = confusion_matrix(baseline_categories, pred_categories, labels=labels)
print("Confusion Matrix:\n", cm)

precisions = {}
recalls = {}
f1_scores = {}

for i, label in enumerate(labels):
    true_positive = cm[i,i]
    false_positive = cm[:, i].sum() - true_positive
    false_negative = cm[i, :].sum() - true_positive

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    precisions[label] = precision
    recalls[label] = recall
    f1_scores[label] = f1

print("\nPer-class Precision:", precisions)
print("Per-class Recall:", recalls)
print("Per-class F1 score:", f1_scores)

# Calculate weighted metrics
total_samples = len(baseline_categories)
weighted_precision = sum(precisions[label] * cm[i, :].sum() for i, label in enumerate(labels)) / total_samples
weighted_recall = sum(recalls[label] * cm[i, :].sum() for i, label in enumerate(labels)) / total_samples
weighted_f1 = sum(f1_scores[label] * cm[i, :].sum() for i, label in enumerate(labels)) / total_samples

print("\nOverall Metrics:")
print("Weighted Precision:", weighted_precision)
print("Weighted Recall:", weighted_recall)
print("Weighted F1 Score:", weighted_f1)

# Calculate accuracy
accuracy = accuracy_score(baseline_categories, pred_categories)
print("Accuracy Score:", accuracy)