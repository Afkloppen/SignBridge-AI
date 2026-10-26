
from pathlib import Path
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

DATA_FILE = Path("data/processed/landmarks.npz")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "signbridge_random_forest.joblib"
LABELS_FILE = MODEL_DIR / "random_forest_labels.json"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_FILE}")

data = np.load(DATA_FILE, allow_pickle=False)
X = data["X"].astype(np.float32)
y = data["y"].astype(str)
splits = data["splits"].astype(str)

# Keep only labels represented in the training set.
train_labels = sorted(set(y[splits == "train"]))
keep = np.isin(y, train_labels)
X, y, splits = X[keep], y[keep], splits[keep]

# Flatten each 16-frame sequence into one feature vector.
X = X.reshape(len(X), -1)

train_mask = splits == "train"
val_mask = splits == "val"
test_mask = splits == "test"

if not train_mask.any() or not val_mask.any() or not test_mask.any():
    raise ValueError("Train, validation, and test splits must all exist.")

# Fit preprocessing only on training data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X[train_mask])
X_val = scaler.transform(X[val_mask])
X_test = scaler.transform(X[test_mask])

y_train = y[train_mask]
y_val = y[val_mask]
y_test = y[test_mask]

print("Training videos:", len(y_train))
print("Validation videos:", len(y_val))
print("Test videos:", len(y_test))
print("Classes:", train_labels)

best_model = None
best_val_accuracy = -1.0

# Compare a few model sizes using validation accuracy.
for trees in [200, 400]:
    model = RandomForestClassifier(
        n_estimators=trees,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)

    print(f"\nTrees: {trees} | Validation accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = model

# Evaluate the selected model on the held-out test split.
test_pred = best_model.predict(X_test)

print("\n========== RANDOM FOREST TEST RESULTS ==========")
print("Test accuracy:", round(accuracy_score(y_test, test_pred), 4))
print(classification_report(
    y_test,
    test_pred,
    labels=train_labels,
    zero_division=0,
))
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_test, test_pred, labels=train_labels))
print("Class order:", train_labels)

joblib.dump(
    {
        "model": best_model,
        "scaler": scaler,
        "labels": train_labels,
        "num_frames": int(data["X"].shape[1]),
        "input_size": int(data["X"].shape[2]),
    },
    MODEL_FILE,
)

LABELS_FILE.write_text(
    json.dumps(train_labels, indent=2),
    encoding="utf-8",
)

print("\nModel saved to:", MODEL_FILE)
print("Labels saved to:", LABELS_FILE)
