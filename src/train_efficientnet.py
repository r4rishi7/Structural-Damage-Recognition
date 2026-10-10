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
from src.efficientnet_preprocessing import restore_rgb_images


MODEL_PATH = Path("models/efficientnetb0.keras")
RESULT_DIR = Path("results/experiments/efficientnetb0")

BATCH_SIZE = 8
EPOCHS = 10
RANDOM_STATE = 42


class EfficientNetGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        images, labels = self.generator[index]
        images = restore_rgb_images(images)
        return images, labels

    def on_epoch_end(self):
        self.generator.on_epoch_end()


def build_model():
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(
        2, activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
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

    print("Training batches:", len(train_gen))
    print("Validation batches:", len(val_gen))
    print("Test batches:", len(test_gen))

    print("Building EfficientNetB0...")
    model = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(
            str(RESULT_DIR / "history.csv")
        ),
    ]

    print("Starting EfficientNetB0 training...")
    start_time = time.perf_counter()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    training_seconds = time.perf_counter() - start_time

    # Load the best validation checkpoint.
    model = tf.keras.models.load_model(str(MODEL_PATH))

    print("Evaluating best checkpoint on test data...")

    probabilities = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    y_true = np.argmax(
        test_base.y[test_base.current_indices], axis=1
    )

    # Positive class 1 = Undamaged, matching the
    # original evaluation metric convention.
    results = {
        "model": "EfficientNetB0",
        "baseline_accuracy": 0.76780822,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, pos_label=1
        )),
        "recall": float(recall_score(
            y_true, y_pred, pos_label=1
        )),
        "f1": float(f1_score(
            y_true, y_pred, pos_label=1
        )),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "training_seconds": training_seconds,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history.history["loss"]),
        "learning_rate": 0.0001,
        "random_state": RANDOM_STATE,
        "backbone_frozen": True,
        "preprocessing": "Caffe BGR mean reversal to RGB",
    }

    result_path = RESULT_DIR / "results.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nFinal EfficientNetB0 Results")
    print("----------------------------")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    print(f"Training seconds: {training_seconds:.1f}")
    print(f"Results saved: {result_path}")


if __name__ == "__main__":
    main()
    