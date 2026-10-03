from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_X_PATH = (
    PROJECT_ROOT
    / "data"
    / "task2_damage_state_2"
    / "task2_X_train.npy"
)

TRAIN_Y_PATH = (
    PROJECT_ROOT
    / "data"
    / "task2_damage_state_1"
    / "task2_y_train.npy"
)

TEST_X_PATH = (
    PROJECT_ROOT
    / "data"
    / "task2_damage_state_1"
    / "task2_X_test.npy"
)

TEST_Y_PATH = (
    PROJECT_ROOT
    / "data"
    / "task2_damage_state_1"
    / "task2_y_test.npy"
)


def load_training_arrays():
    """Load training arrays using memory mapping for images."""
    X = np.load(TRAIN_X_PATH, mmap_mode="r")
    y = np.load(TRAIN_Y_PATH)

    return X, y


def load_test_arrays():
    """Load test arrays using memory mapping for images."""
    X = np.load(TEST_X_PATH, mmap_mode="r")
    y = np.load(TEST_Y_PATH)

    return X, y


def get_train_validation_indices(
    y,
    validation_size=0.1,
    random_state=42,
):
    """Create stratified train/validation indices."""
    labels = np.argmax(y, axis=1)
    indices = np.arange(len(y))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )

    return train_indices, validation_indices


def get_class_labels(y):
    """Convert one-hot labels to integer class labels."""
    return np.argmax(y, axis=1)


if __name__ == "__main__":
    X_train, y_train = load_training_arrays()
    X_test, y_test = load_test_arrays()

    train_indices, validation_indices = get_train_validation_indices(
        y_train
    )

    print("Training dataset:", X_train.shape)
    print("Test dataset:", X_test.shape)

    print("Training samples:", len(train_indices))
    print("Validation samples:", len(validation_indices))

    train_labels = get_class_labels(y_train)

    print(
        "Training class distribution:",
        np.unique(
            train_labels[train_indices],
            return_counts=True,
        ),
    )

    print(
        "Validation class distribution:",
        np.unique(
            train_labels[validation_indices],
            return_counts=True,
        ),
    )