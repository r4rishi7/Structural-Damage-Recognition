from pathlib import Path

import numpy as np
import tensorflow as tf

from src.batch_generator import create_generators
from src.evaluate import evaluate_model


MODEL_PATH = Path("models/inceptionv3.keras")


def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Creating test generator...")
    _, _, test_generator = create_generators(
        validation_size=0.1,
        batch_size=32,
        random_state=42,
    )

    print("Generating predictions...")
    y_true = []
    y_pred = []

    for images, labels in test_generator:
        predictions = model.predict(images, verbose=0)

        y_true.append(labels)
        y_pred.append(predictions)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    print("\nFinal Test Evaluation")
    print("=====================")

    evaluate_model(y_true, y_pred)


if __name__ == "__main__":
    main()