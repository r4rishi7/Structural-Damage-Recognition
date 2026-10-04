import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/mobilenetv1.keras"

X_TEST_PATH = "DATA/task2_damage_state_1/task2_X_test.npy"
Y_TEST_PATH = "DATA/task2_damage_state_1/task2_y_test.npy"

RESULTS_DIR = "results/classical_ml"
RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "mobilenetv1_results.json"
)


# -----------------------------
# Load model
# -----------------------------
print("Loading MobileNetV1...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# -----------------------------
# Load untouched test set
# -----------------------------
print("\nLoading test data...")

X_test = np.load(X_TEST_PATH, mmap_mode="r")
y_test_onehot = np.load(Y_TEST_PATH, mmap_mode="r")

y_test = np.argmax(y_test_onehot, axis=1)

print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)


# -----------------------------
# Prediction
# -----------------------------
print("\nRunning predictions...")

y_prob = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(y_prob, axis=1)


# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

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

cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)


# -----------------------------
# Print results
# -----------------------------
print("\n==============================")
print("MobileNetV1 Test Results")
print("==============================")

print(f"Accuracy :  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision:  {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   :  {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score :  {f1:.4f} ({f1 * 100:.2f}%)")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)


# -----------------------------
# Save results
# -----------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)

results = {
    "model": "MobileNetV1",
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist()
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to:")
print(RESULTS_PATH)