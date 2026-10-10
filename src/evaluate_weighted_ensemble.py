import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.batch_generator import create_generators


RESULT_DIR = Path("results/experiments/weighted_ensemble")
BATCH_SIZE = 16
RANDOM_STATE = 42

# Candidate weights for MobileNetV1.
WEIGHTS = np.arange(0.0, 1.01, 0.1)


def predict_model(model, generator):
    return model.predict(generator, verbose=1)


def ensemble_predictions(mobile_probs, inception_probs, weight):
    probabilities = (
        weight * mobile_probs
        + (1.0 - weight) * inception_probs
    )
    return np.argmax(probabilities, axis=1)


def main():
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    _, val_gen, test_gen = create_generators(
        validation_size=0.1,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Loading original baseline models...")

    mobilenet = tf.keras.models.load_model(
        "models/mobilenetv1.keras"
    )
    inception = tf.keras.models.load_model(
        "models/inceptionv3.keras"
    )

    start_time = time.perf_counter()

    print("Predicting validation probabilities...")

    mobile_val = predict_model(mobilenet, val_gen)
    inception_val = predict_model(inception, val_gen)

    y_val = np.argmax(
        val_gen.y[val_gen.current_indices],
        axis=1,
    )

    search_results = []

    for weight in WEIGHTS:
        predictions = ensemble_predictions(
            mobile_val, inception_val, weight
        )

        accuracy = accuracy_score(y_val, predictions)

        search_results.append({
            "mobilenet_weight": round(float(weight), 2),
            "inception_weight": round(float(1 - weight), 2),
            "validation_accuracy": float(accuracy),
        })

    # Choose the best weight using validation data only.
    best = max(
        search_results,
        key=lambda item: item["validation_accuracy"],
    )

    best_weight = best["mobilenet_weight"]

    print("Selected MobileNetV1 weight:", best_weight)
    print("Best validation accuracy:", best["validation_accuracy"])

    print("Predicting test probabilities...")

    mobile_test = predict_model(mobilenet, test_gen)
    inception_test = predict_model(inception, test_gen)

    y_test = np.argmax(
        test_gen.y[test_gen.current_indices],
        axis=1,
    )

    y_pred = ensemble_predictions(
        mobile_test, inception_test, best_weight
    )

    elapsed_seconds = time.perf_counter() - start_time

    results = {
        "model": "MobileNetV1 + InceptionV3 weighted ensemble",
        "mobilenet_weight": best_weight,
        "inception_weight": round(1 - best_weight, 2),
        "validation_accuracy": best["validation_accuracy"],
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, pos_label=1)
        ),
        "recall": float(
            recall_score(y_test, y_pred, pos_label=1)
        ),
        "f1": float(
            f1_score(y_test, y_pred, pos_label=1)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).tolist(),
        "elapsed_seconds": elapsed_seconds,
        "random_state": RANDOM_STATE,
        "weight_search": search_results,
    }

    with open(
        RESULT_DIR / "results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    print("\nWeighted Ensemble Results")
    print("-------------------------")
    print("MobileNetV1 weight:", best_weight)
    print("InceptionV3 weight:", round(1 - best_weight, 2))
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    print("Confusion matrix:", results["confusion_matrix"])
    print(f"Elapsed seconds: {elapsed_seconds:.1f}")


if __name__ == "__main__":
    main()
    