
from pathlib import Path
import gc
import time

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import vstack

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.pipeline import Pipeline


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

JIGSAW_PATH = BASE_DIR / "data" / "jigsaw_dataset" / "train.csv"
CUSTOM_PATH = BASE_DIR / "data" / "comments.csv"

MODEL_DIR = BASE_DIR / "models"

BINARY_MODEL_PATH = MODEL_DIR / "classifier.joblib"
MULTILABEL_MODEL_PATH = MODEL_DIR / "multilabel_classifier.joblib"

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

RANDOM_STATE = 42

# Optimized for your current 16 GB RAM system.
MAX_FEATURES = 40000

print("=" * 55)
print(" MULTILINGUAL CYBERBULLYING DETECTION TRAINING")
print("=" * 55)


# ==========================================
# 2. LOAD JIGSAW DATASET
# ==========================================

print("\n[1/9] Loading Jigsaw dataset...")

if not JIGSAW_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {JIGSAW_PATH}")

jigsaw = pd.read_csv(JIGSAW_PATH)

required_columns = ["comment_text"] + LABEL_COLUMNS

if not all(col in jigsaw.columns for col in required_columns):
    raise ValueError("Jigsaw dataset is missing required columns.")

jigsaw = jigsaw.dropna(subset=["comment_text"]).copy()

jigsaw["comment_text"] = (
    jigsaw["comment_text"].astype(str).str.strip()
)

jigsaw = jigsaw[jigsaw["comment_text"] != ""]

for col in LABEL_COLUMNS:
    jigsaw[col] = pd.to_numeric(jigsaw[col], errors="coerce")

jigsaw = jigsaw.dropna(subset=LABEL_COLUMNS)

if not jigsaw[LABEL_COLUMNS].isin([0, 1]).all().all():
    raise ValueError("Jigsaw labels must contain only 0 and 1.")

jigsaw[LABEL_COLUMNS] = jigsaw[LABEL_COLUMNS].astype(np.int8)

jigsaw = jigsaw.reset_index(drop=True)

print("Jigsaw comments:", len(jigsaw))


# ==========================================
# 3. LOAD CUSTOM MULTILINGUAL DATA
# ==========================================

print("\n[2/9] Loading custom dataset...")

if not CUSTOM_PATH.exists():
    raise FileNotFoundError(f"Custom dataset not found: {CUSTOM_PATH}")

custom = pd.read_csv(CUSTOM_PATH)

if not {"text", "label"}.issubset(custom.columns):
    raise ValueError("comments.csv needs text and label columns.")

custom = custom.dropna(subset=["text", "label"]).copy()

custom["text"] = custom["text"].astype(str).str.strip()

custom["label"] = pd.to_numeric(
    custom["label"], errors="coerce"
)

custom = custom.dropna(subset=["label"])

custom = custom[custom["text"] != ""]

if not custom["label"].isin([0, 1]).all():
    raise ValueError("Custom labels must be 0 or 1.")

custom["label"] = custom["label"].astype(np.int8)

if set(custom["label"].unique()) != {0, 1}:
    raise ValueError("Custom data needs both classes: 0 and 1.")

custom = custom.reset_index(drop=True)

print("Custom comments:", len(custom))


# ==========================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

print("\n[3/9] Splitting datasets...")

jigsaw_binary = jigsaw[LABEL_COLUMNS].max(axis=1)

indices = np.arange(len(jigsaw))

train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=jigsaw_binary,
)

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.1765,
    random_state=RANDOM_STATE,
    stratify=jigsaw_binary.iloc[train_val_idx],
)

train_df = jigsaw.iloc[train_idx].reset_index(drop=True)
val_df = jigsaw.iloc[val_idx].reset_index(drop=True)
test_df = jigsaw.iloc[test_idx].reset_index(drop=True)

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Testing rows:", len(test_df))


# ==========================================
# 5. TF-IDF FEATURE EXTRACTION
# ==========================================

print("\n[4/9] Extracting TF-IDF features...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    max_features=MAX_FEATURES,
    sublinear_tf=True,
    dtype=np.float32,
)

# Fit vocabulary on training comments and custom data only.

fit_text = pd.concat(
    [
        train_df["comment_text"],
        custom["text"],
    ],
    ignore_index=True,
)

vectorizer.fit(fit_text)

del fit_text
gc.collect()

print("Vocabulary size:", len(vectorizer.vocabulary_))

X_train = vectorizer.transform(train_df["comment_text"])
X_val = vectorizer.transform(val_df["comment_text"])
X_test = vectorizer.transform(test_df["comment_text"])

X_custom = vectorizer.transform(custom["text"])

print("Training matrix:", X_train.shape)
print("Validation matrix:", X_val.shape)
print("Testing matrix:", X_test.shape)

# Multilabel targets

y_train_multi = train_df[LABEL_COLUMNS].to_numpy(dtype=np.int8)
y_val_multi = val_df[LABEL_COLUMNS].to_numpy(dtype=np.int8)
y_test_multi = test_df[LABEL_COLUMNS].to_numpy(dtype=np.int8)

# Binary targets

y_train_binary = y_train_multi.max(axis=1)
y_val_binary = y_val_multi.max(axis=1)
y_test_binary = y_test_multi.max(axis=1)

# Add custom multilingual samples to binary training only.

X_binary_train = vstack(
    [X_train, X_custom],
    format="csr",
    dtype=np.float32,
)

y_binary_train = np.concatenate(
    [y_train_binary, custom["label"].to_numpy()]
)

print("Binary training matrix:", X_binary_train.shape)


# ==========================================
# 6. DEFINE ALGORITHMS
# ==========================================

print("\n[5/9] Preparing classification algorithms...")

binary_algorithms = {
    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=0.5
    ),
}

multilabel_algorithms = {
    "Logistic Regression": OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        n_jobs=1,
    ),

    "Linear SVM": OneVsRestClassifier(
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        n_jobs=1,
    ),

    "Multinomial Naive Bayes": OneVsRestClassifier(
        MultinomialNB(alpha=0.5),
        n_jobs=1,
    ),
}


# ==========================================
# 7. BINARY CLASSIFICATION
# ==========================================

print("\n========== BINARY CLASSIFICATION ==========")

best_binary_model = None
best_binary_name = None
best_binary_f1 = -1

for name, algorithm in binary_algorithms.items():

    start_time = time.time()

    print(f"\nTraining {name}...")

    algorithm.fit(X_binary_train, y_binary_train)

    predictions = algorithm.predict(X_val)

    precision = precision_score(
        y_val_binary, predictions, zero_division=0
    )

    recall = recall_score(
        y_val_binary, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val_binary, predictions, zero_division=0
    )

    accuracy = accuracy_score(y_val_binary, predictions)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("Time:", round(time.time() - start_time, 2), "seconds")

    if f1 > best_binary_f1:
        best_binary_f1 = f1
        best_binary_model = clone(algorithm)
        best_binary_name = name

    del algorithm
    gc.collect()

print("\nBest binary algorithm:", best_binary_name)


# ==========================================
# 8. MULTILABEL CLASSIFICATION
# ==========================================

print("\n========== MULTILABEL CLASSIFICATION ==========")

best_multilabel_model = None
best_multilabel_name = None
best_multilabel_f1 = -1

for name, algorithm in multilabel_algorithms.items():

    start_time = time.time()

    print(f"\nTraining multilabel {name}...")

    algorithm.fit(X_train, y_train_multi)

    predictions = algorithm.predict(X_val)

    micro_f1 = f1_score(
        y_val_multi,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_val_multi,
        predictions,
        average="macro",
        zero_division=0,
    )

    print("Micro F1:", round(micro_f1, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Time:", round(time.time() - start_time, 2), "seconds")

    if macro_f1 > best_multilabel_f1:
        best_multilabel_f1 = macro_f1
        best_multilabel_model = clone(algorithm)
        best_multilabel_name = name

    del algorithm
    gc.collect()

print("\nBest multilabel algorithm:", best_multilabel_name)


# ==========================================
# 9. FINAL TRAINING AND EVALUATION
# ==========================================

print("\n[6/9] Retraining selected models...")

# Combine training and validation data.
# Keep the final test set completely untouched.

X_train_val = vstack(
    [X_train, X_val],
    format="csr",
    dtype=np.float32,
)

y_train_val_multi = np.vstack(
    [y_train_multi, y_val_multi]
)

y_train_val_binary = np.concatenate(
    [y_train_binary, y_val_binary]
)

X_binary_train_val = vstack(
    [X_train_val, X_custom],
    format="csr",
    dtype=np.float32,
)

y_binary_train_val = np.concatenate(
    [y_train_val_binary, custom["label"].to_numpy()]
)

# Final binary model

print("Final binary model:", best_binary_name)

best_binary_model.fit(
    X_binary_train_val,
    y_binary_train_val,
)

binary_predictions = best_binary_model.predict(X_test)

print("\nFINAL BINARY TEST RESULTS")

print("Accuracy:", round(
    accuracy_score(y_test_binary, binary_predictions), 4
))

print("Precision:", round(
    precision_score(y_test_binary, binary_predictions, zero_division=0), 4
))

print("Recall:", round(
    recall_score(y_test_binary, binary_predictions, zero_division=0), 4
))

print("F1:", round(
    f1_score(y_test_binary, binary_predictions, zero_division=0), 4
))

print(classification_report(
    y_test_binary,
    binary_predictions,
    target_names=["Non-toxic", "Toxic"],
    zero_division=0,
))

# Final multilabel model

print("\nFinal multilabel model:", best_multilabel_name)

best_multilabel_model.fit(
    X_train_val,
    y_train_val_multi,
)

multi_predictions = best_multilabel_model.predict(X_test)

print("\nFINAL MULTILABEL TEST RESULTS")

print("Micro F1:", round(
    f1_score(
        y_test_multi,
        multi_predictions,
        average="micro",
        zero_division=0,
    ), 4
))

print("Macro F1:", round(
    f1_score(
        y_test_multi,
        multi_predictions,
        average="macro",
        zero_division=0,
    ), 4
))

print("\nCategory-wise report:")

print(classification_report(
    y_test_multi,
    multi_predictions,
    target_names=LABEL_COLUMNS,
    zero_division=0,
))


# ==========================================
# 10. SAVE MODELS
# ==========================================

print("\n[7/9] Saving trained models...")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

binary_pipeline = Pipeline([
    ("tfidf", vectorizer),
    ("classifier", best_binary_model),
])

multilabel_pipeline = Pipeline([
    ("tfidf", vectorizer),
    ("classifier", best_multilabel_model),
])

joblib.dump(
    binary_pipeline,
    BINARY_MODEL_PATH,
    compress=3,
)

joblib.dump(
    multilabel_pipeline,
    MULTILABEL_MODEL_PATH,
    compress=3,
)


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("          TRAINING COMPLETED!")
print("=" * 55)

print("\nJigsaw dataset:", len(jigsaw))
print("Custom dataset:", len(custom))

print("\nBest binary algorithm:", best_binary_name)
print("Best validation F1:", round(best_binary_f1, 4))

print("\nBest multilabel algorithm:", best_multilabel_name)
print("Best validation macro F1:", round(best_multilabel_f1, 4))

print("\nSaved models:")

print(BINARY_MODEL_PATH)
print(MULTILABEL_MODEL_PATH)

print("\nToxicity categories:")

for label in LABEL_COLUMNS:
    print("-", label)

print("\nBoth models are saved successfully!")
print("Next: Integrate the models into Flask.")
