import os
import tensorflow as tf

from src.models import build_inceptionv3
from src.batch_generator import create_generators


# =========================
# Configuration
# =========================

MODEL_PATH = "models/inceptionv3.keras"

BATCH_SIZE = 16
EPOCHS = 10
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42


# =========================
# Main training
# =========================

def main():

    print("Creating train/validation/test generators...")

    (
        train_generator,
        validation_generator,
        test_generator,
    ) = create_generators(
        validation_size=VALIDATION_SIZE,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Training batches   : {len(train_generator)}")
    print(f"Validation batches : {len(validation_generator)}")
    print(f"Test batches       : {len(test_generator)}")

    # =========================
    # Build model
    # =========================

    print("\nBuilding InceptionV3 model...")

    model = build_inceptionv3(
        input_shape=(224, 224, 3),
        num_classes=2,
    )

    model.summary()

    # =========================
    # Directories
    # =========================

    os.makedirs("models", exist_ok=True)
    os.makedirs("results/training", exist_ok=True)

    # =========================
    # Callbacks
    # =========================

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            "results/training/history_final.csv"
        ),
    ]

    # =========================
    # Training
    # =========================

    print("\nStarting training...\n")

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # =========================
    # FINAL TEST EVALUATION
    # =========================

    print("\nEvaluating on untouched test set...")

    loss, accuracy = model.evaluate(
        test_generator,
        verbose=1,
    )

    print("\n==============================")
    print("Final Test Results")
    print("==============================")
    print(f"Test Loss     : {loss:.4f}")
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
    print("==============================")

    model.save(MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(
        "Training history saved to: "
        "results/training/history_final.csv"
    )


if __name__ == "__main__":
    main()