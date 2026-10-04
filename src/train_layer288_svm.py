import json
import os
import time

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


FEATURE_FILE = "results/classical_ml/inceptionv3_layer288_features.npz"
OUTPUT_DIR = "results/classical_ml"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "inceptionv3_layer288_svm_results.json"
)

print("Loading layer-288 features...")

data = np.load(FEATURE_FILE)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nTraining RBF-SVM...")
print("C = 1.0")
print("gamma = 0.001")

start_time = time.time()

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma=0.001,
    cache_size=4096
)

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds.")

print("\nEvaluating validation set...")

val_pred = model.predict(X_val)

val_accuracy = accuracy_score(y_val, val_pred)
val_precision = precision_score(y_val, val_pred, zero_division=0)
val_recall = recall_score(y_val, val_pred, zero_division=0)
val_f1 = f1_score(y_val, val_pred, zero_division=0)
val_cm = confusion_matrix(y_val, val_pred)

print(f"Validation Accuracy:  {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall:    {val_recall:.4f}")
print(f"Validation F1:        {val_f1:.4f}")
print("Validation Confusion Matrix:")
print(val_cm)

print("\nEvaluating test set...")

test_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred, zero_division=0)
test_recall = recall_score(y_test, test_pred, zero_division=0)
test_f1 = f1_score(y_test, test_pred, zero_division=0)
test_cm = confusion_matrix(y_test, test_pred)

print(f"Test Accuracy:  {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1:        {test_f1:.4f}")
print("Test Confusion Matrix:")
print(test_cm)

results = {
    "model": "InceptionV3 Layer 288 + RBF-SVM",
    "feature_layer": 288,
    "feature_layer_name": "activation_90",
    "feature_shape": [5, 5, 384],
    "flattened_features": 9600,
    "kernel": "rbf",
    "C": 1.0,
    "gamma": 0.001,
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "training_time_seconds": training_time,
    "validation": {
        "accuracy": float(val_accuracy),
        "precision": float(val_precision),
        "recall": float(val_recall),
        "f1": float(val_f1),
        "confusion_matrix": val_cm.tolist(),
    },
    "test": {
        "accuracy": float(test_accuracy),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1": float(test_f1),
        "confusion_matrix": test_cm.tolist(),
        "classification_report": classification_report(
            y_test,
            test_pred,
            output_dict=True,
            zero_division=0,
        ),
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", OUTPUT_FILE)