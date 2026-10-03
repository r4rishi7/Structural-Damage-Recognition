import os
import numpy as np
import tensorflow as tf

from src.models import build_inceptionv3


# =========================
# Configuration
# =========================

TRAIN_X_PATH = "DATA/task2_damage_state_2/task2_X_train.npy"
TRAIN_Y_PATH = "DATA/task2_damage_state_1/task2_y_train.npy"

TEST_X_PATH = "DATA/task2_damage_state_1/task2_X_test.npy"
TEST_Y_PATH = "DATA/task2_damage_state_1/task2_y_test.npy"

MODEL_PATH = "models/inceptionv3.keras"

BATCH_SIZE = 16
EPOCHS = 10


# =========================
# Data generator
# =========================

class NumpyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_path, y_path, batch_size=16, shuffle=False):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(len(self.y))

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_x = np.asarray(self.x[batch_indices], dtype=np.float32)
        batch_y = np.asarray(self.y[batch_indices], dtype=np.float32)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =========================
# Main training
# =========================

def main():

    print("Loading dataset...")

    train_generator = NumpyDataGenerator(
        TRAIN_X_PATH,
        TRAIN_Y_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_generator = NumpyDataGenerator(
        TEST_X_PATH,
        TEST_Y_PATH,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"Training samples: {len(train_generator.y)}")
    print(f"Test samples: {len(test_generator.y)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training batches: {len(train_generator)}")
    print(f"Test batches: {len(test_generator)}")

    print("\nBuilding InceptionV3 model...")

    model = build_inceptionv3(
        input_shape=(224, 224, 3),
        num_classes=2
    )

    model.summary()

    # =========================
    # Callbacks
    # =========================

    os.makedirs("models", exist_ok=True)
    os.makedirs("results/training", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            "results/training/history.csv"
        )
    ]

    # =========================
    # Train
    # =========================

    print("\nStarting training...\n")

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # =========================
    # Final evaluation
    # =========================

    print("\nEvaluating model on test set...")

    loss, accuracy = model.evaluate(
        test_generator,
        verbose=1
    )

    print("\n==============================")
    print("Final Test Results")
    print("==============================")
    print(f"Test Loss     : {loss:.4f}")
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
    print("==============================")

    # Save final model as well
    model.save(MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Training history saved to: results/training/history.csv")


if __name__ == "__main__":
    main()