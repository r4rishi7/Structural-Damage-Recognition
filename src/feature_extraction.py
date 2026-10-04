from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_X_PATH = (
    PROJECT_ROOT
    / "DATA"
    / "task2_damage_state_2"
    / "task2_X_train.npy"
)

TRAIN_Y_PATH = (
    PROJECT_ROOT
    / "DATA"
    / "task2_damage_state_1"
    / "task2_y_train.npy"
)

TEST_X_PATH = (
    PROJECT_ROOT
    / "DATA"
    / "task2_damage_state_1"
    / "task2_X_test.npy"
)

TEST_Y_PATH = (
    PROJECT_ROOT
    / "DATA"
    / "task2_damage_state_1"
    / "task2_y_test.npy"
)

MODEL_PATH = PROJECT_ROOT / "models" / "inceptionv3.keras"

OUTPUT_DIR = PROJECT_ROOT / "results" / "classical_ml"

BATCH_SIZE = 32
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42


# ============================================================
# Feature extraction
# ============================================================

def extract_features(feature_model, X, batch_size=32):
    """Extract 2048-D InceptionV3 features in batches."""

    features = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))

        batch = np.asarray(X[start:end], dtype=np.float32)

        batch_features = feature_model.predict(
            batch,
            verbose=0
        )

        features.append(batch_features)

        print(
            f"Processed {end}/{len(X)} images",
            flush=True
        )

    return np.concatenate(features, axis=0)


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading final InceptionV3 model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    # Extract features from the GlobalAveragePooling2D layer.
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("global_average_pooling2d").output,
    )

    print(
        "Feature extractor output shape:",
        feature_model.output_shape
    )

    # --------------------------------------------------------
    # Load data using memory mapping
    # --------------------------------------------------------

    print("\nLoading dataset...")

    X_train = np.load(
        TRAIN_X_PATH,
        mmap_mode="r"
    )

    y_train = np.load(TRAIN_Y_PATH)

    X_test = np.load(
        TEST_X_PATH,
        mmap_mode="r"
    )

    y_test = np.load(TEST_Y_PATH)

    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    print("Training images:", X_train.shape)
    print("Training labels:", y_train.shape)

    print("Test images:", X_test.shape)
    print("Test labels:", y_test.shape)

    # --------------------------------------------------------
    # Extract 2048-D features
    # --------------------------------------------------------

    print("\nExtracting training features...")

    train_features = extract_features(
        feature_model,
        X_train,
        batch_size=BATCH_SIZE,
    )

    print("\nExtracting test features...")

    test_features = extract_features(
        feature_model,
        X_test,
        batch_size=BATCH_SIZE,
    )

    print("\nFeature extraction completed.")

    print("Train features:", train_features.shape)
    print("Test features:", test_features.shape)

    # --------------------------------------------------------
    # Reproduce the same 90/10 stratified split
    # --------------------------------------------------------

    indices = np.arange(len(y_train_labels))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_labels,
    )

    X_train_features = train_features[train_indices]
    X_val_features = train_features[validation_indices]

    y_train_final = y_train_labels[train_indices]
    y_val_final = y_train_labels[validation_indices]

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / "inceptionv3_features.npz"

    np.savez_compressed(
        output_path,
        X_train=X_train_features,
        y_train=y_train_final,
        X_val=X_val_features,
        y_val=y_val_final,
        X_test=test_features,
        y_test=y_test_labels,
    )

    print("\nSaved:", output_path)

    print("\nFinal feature shapes:")
    print("X_train:", X_train_features.shape)
    print("y_train:", y_train_final.shape)
    print("X_val  :", X_val_features.shape)
    print("y_val  :", y_val_final.shape)
    print("X_test :", test_features.shape)
    print("y_test :", y_test_labels.shape)


if __name__ == "__main__":
    main()