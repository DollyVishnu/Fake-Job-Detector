import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Load Dataset
# ==========================================

DATASET_PATH = "dataset/DataSet.csv"

print("\nLoading dataset...")

data = pd.read_csv(DATASET_PATH)

print(f"Total rows: {len(data)}")
print(f"Total columns: {len(data.columns)}")


# ==========================================
# 2. Check Required Columns
# ==========================================

required_columns = [
    "title",
    "description",
    "fraudulent"
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")

    for column in missing_columns:
        print("-", column)

    print("\nRequired columns:")
    print(required_columns)

    raise SystemExit


# ==========================================
# 3. Clean Missing Values
# ==========================================

data = data.fillna("")


# ==========================================
# 4. Remove Duplicate Jobs
# ==========================================

before = len(data)

data = data.drop_duplicates(
    subset=["title", "description"]
)

after = len(data)

print(f"\nDuplicate rows removed: {before - after}")


# ==========================================
# 5. Combine Text
# ==========================================

data["text"] = (
    data["title"].astype(str)
    + " "
    + data["description"].astype(str)
)


# ==========================================
# 6. Remove Empty Text
# ==========================================

data = data[
    data["text"].str.strip().str.len() > 0
]


# ==========================================
# 7. Prepare X and y
# ==========================================

X = data["text"]

labels = (
    data["fraudulent"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"t": "1", "f": "0", "true": "1", "false": "0"})
)

y = pd.to_numeric(labels, errors="coerce")

# Remove invalid labels
valid_rows = y.notna()

X = X[valid_rows]
y = y[valid_rows].astype(int)


# ==========================================
# 8. Dataset Information
# ==========================================

print("\nClass Distribution:")

print(
    y.value_counts()
    .sort_index()
)

print("\nClass Percentages:")

print(
    (y.value_counts(normalize=True) * 100)
    .sort_index()
    .round(2)
)


# ==========================================
# 9. Check Dataset Size
# ==========================================

if len(y) < 50:

    print("\n⚠️ WARNING")
    print("Dataset contains fewer than 50 valid rows.")
    print("Model evaluation may not be reliable.")
    print("Use a larger dataset for meaningful results.")


if y.nunique() < 2:

    raise ValueError(
        "Dataset must contain both genuine and fraudulent jobs."
    )


# ==========================================
# 10. TF-IDF Vectorization
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=1
)

X_vectorized = vectorizer.fit_transform(X)

print(
    f"TF-IDF feature shape: {X_vectorized.shape}"
)


# ==========================================
# 11. Train / Test Split
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# ==========================================
# 12. Train Logistic Regression
# ==========================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ==========================================
# 13. Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 14. Evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


# ==========================================
# 15. Display Results
# ==========================================

print("\n" + "=" * 50)

print("MODEL PERFORMANCE")

print("=" * 50)

print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Precision          : {precision:.4f}"
)

print(
    f"Recall             : {recall:.4f}"
)

print(
    f"F1 Score           : {f1:.4f}"
)

print(
    f"Balanced Accuracy  : {balanced_accuracy:.4f}"
)


# ==========================================
# 16. Classification Report
# ==========================================

print("\n" + "=" * 50)

print("CLASSIFICATION REPORT")

print("=" * 50)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Genuine",
            "Fake"
        ],
        zero_division=0
    )
)


# ==========================================
# 17. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\n" + "=" * 50)

print("CONFUSION MATRIX")

print("=" * 50)

print(
    "                 Predicted"
)

print(
    "              Genuine  Fake"
)

print(
    f"Actual Genuine   {cm[0][0]:4d}   {cm[0][1]:4d}"
)

print(
    f"Actual Fake      {cm[1][0]:4d}   {cm[1][1]:4d}"
)


# ==========================================
# 18. Save Model
# ==========================================

print("\nSaving model...")

with open(
    "fake_job_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ==========================================
# 19. Save Vectorizer
# ==========================================

with open(
    "vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


# ==========================================
# 20. Finished
# ==========================================

print("\n" + "=" * 50)

print("✅ MODEL TRAINING COMPLETED")

print("=" * 50)

print(
    "Saved files:"
)

print(
    "✓ fake_job_model.pkl"
)

print(
    "✓ vectorizer.pkl"
)

print("\nYour Flask application can now use the new model.")