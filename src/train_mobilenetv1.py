import os
import csv
import tensorflow as tf

from src.preprocessing import load_training_data, create_train_validation_split
from src.models import build_mobilenetv1


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 32
EPOCHS = 10
RANDOM_STATE = 42

MODEL_PATH = "models/mobilenetv1.keras"
HISTORY_PATH = "results/training/mobilenetv1_history.csv"


# -----------------------------
# Load full training dataset
# -----------------------------
print("Loading training data...")

X, y = load_training_data()

print("Full training data:", X.shape)
print("Full labels:", y.shape)


# -----------------------------
# Create 90/10 stratified split
# -----------------------------
X_train, X_val, y_train, y_val = create_train_validation_split(
    X,
    y,
    validation_size=0.1,
    random_state=RANDOM_STATE
)

print("\nTrain images:", X_train.shape)
print("Validation images:", X_val.shape)
print("Train labels:", y_train.shape)
print("Validation labels:", y_val.shape)


# -----------------------------
# Build MobileNetV1
# -----------------------------
print("\nBuilding MobileNetV1...")

model = build_mobilenetv1(
    input_shape=(224, 224, 3),
    num_classes=2
)

model.summary()


# -----------------------------
# Create output directories
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("results/training", exist_ok=True)


# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]


# -----------------------------
# Train
# -----------------------------
print("\nStarting MobileNetV1 training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# -----------------------------
# Save training history
# -----------------------------
with open(HISTORY_PATH, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "accuracy",
        "loss",
        "val_accuracy",
        "val_loss"
    ])

    for i in range(len(history.history["loss"])):
        writer.writerow([
            i,
            history.history["accuracy"][i],
            history.history["loss"][i],
            history.history["val_accuracy"][i],
            history.history["val_loss"][i]
        ])


print("\nTraining complete.")
print("Model saved to:", MODEL_PATH)
print("History saved to:", HISTORY_PATH)