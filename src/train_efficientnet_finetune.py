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
from src.train_efficientnet import EfficientNetGenerator


MODEL_PATH = Path("models/efficientnetb0_finetuned.keras")
RESULT_DIR = Path("results/experiments/efficientnetb0_finetune")

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-5
RANDOM_STATE = 42


def build_finetuning_model():
    model = tf.keras.models.load_model(
        "models/efficientnetb0.keras"
    )

    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True

    # Freeze all except the final 16 backbone layers.
    for layer in backbone.layers[:-16]:
        layer.trainable = False

    for layer in backbone.layers[-16:]:
        layer.trainable = not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(
        "Trainable backbone layers:",
        sum(layer.trainable for layer in backbone.layers),
    )

    return model


def main():
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_base, val_base, test_base = create_generators(
        validation_size=0.1,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
    )

    train_gen = EfficientNetGenerator(train_base)
    val_gen = EfficientNetGenerator(val_base)
    test_gen = EfficientNetGenerator(test_base)

    model = build_finetuning_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(
            str(RESULT_DIR / "history.csv")
        ),
    ]

    print("Starting EfficientNetB0 fine-tuning...")
    start_time = time.perf_counter()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    training_seconds = time.perf_counter() - start_time

    # Evaluate the best validation checkpoint.
    model = tf.keras.models.load_model(str(MODEL_PATH))

    print("Evaluating fine-tuned model on test data...")

    probabilities = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    y_true = np.argmax(
        test_base.y[test_base.current_indices],
        axis=1,
    )

    results = {
        "model": "EfficientNetB0 fine-tuned",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=1)
        ),
        "f1": float(
            f1_score(y_true, y_pred, pos_label=1)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "training_seconds": training_seconds,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "unfrozen_backbone_layers": 16,
        "random_state": RANDOM_STATE,
        "starting_checkpoint": "models/efficientnetb0.keras",
    }

    with open(
        RESULT_DIR / "results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    print("\nFine-tuning Results")
    print("-------------------")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    print(f"Training seconds: {training_seconds:.1f}")


if __name__ == "__main__":
    main()

    