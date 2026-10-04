from pathlib import Path
import json
import time

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "classical_ml"
    / "inceptionv3_features.npz"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "classical_ml"
)


def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train a model and evaluate it on validation and test sets."""

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    start = time.time()

    print("Training...")
    model.fit(X_train, y_train)

    training_time = time.time() - start

    print(f"Training time: {training_time:.2f} seconds")

    # Validation
    y_val_pred = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(
        y_val,
        y_val_pred,
        average="binary",
        zero_division=0,
    )
    val_recall = recall_score(
        y_val,
        y_val_pred,
        average="binary",
        zero_division=0,
    )
    val_f1 = f1_score(
        y_val,
        y_val_pred,
        average="binary",
        zero_division=0,
    )

    # Test
    print("Evaluating on untouched test set...")

    y_test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(
        y_test,
        y_test_pred,
        average="binary",
        zero_division=0,
    )
    test_recall = recall_score(
        y_test,
        y_test_pred,
        average="binary",
        zero_division=0,
    )
    test_f1 = f1_score(
        y_test,
        y_test_pred,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_test_pred)

    print("\nValidation Results")
    print("------------------")
    print(f"Accuracy : {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall   : {val_recall:.4f}")
    print(f"F1 Score : {val_f1:.4f}")

    print("\nTest Results")
    print("------------")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1 Score : {test_f1:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    results = {
        "model": name,
        "training_time_seconds": training_time,

        "validation": {
            "accuracy": float(val_accuracy),
            "precision": float(val_precision),
            "recall": float(val_recall),
            "f1": float(val_f1),
        },

        "test": {
            "accuracy": float(test_accuracy),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1": float(test_f1),
            "confusion_matrix": cm.tolist(),
        },
    }

    output_file = RESULTS_DIR / f"{name.lower().replace(' ', '_')}_results.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results: {output_file}")

    return results


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading extracted InceptionV3 features...")

    data = np.load(FEATURE_PATH)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    print("\nFeature shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val  :", X_val.shape)
    print("y_val  :", y_val.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    # ========================================================
    # 1. KNN
    # ========================================================

    knn = Pipeline([
        ("scaler", StandardScaler()),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1,
            ),
        ),
    ])

    evaluate_model(
        "KNN",
        knn,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    # ========================================================
    # 2. Logistic Regression
    # ========================================================

    logistic_regression = Pipeline([
        ("scaler", StandardScaler()),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    evaluate_model(
        "Logistic Regression",
        logistic_regression,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    # ========================================================
    # 3. RBF-SVM
    # ========================================================

    rbf_svm = Pipeline([
        ("scaler", StandardScaler()),
        (
            "rbf_svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
            ),
        ),
    ])

    evaluate_model(
        "RBF-SVM",
        rbf_svm,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()