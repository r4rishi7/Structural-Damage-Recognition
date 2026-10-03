from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


# Project paths
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


def load_training_data():
    """Load the training images and one-hot encoded labels."""
    X = np.load(TRAIN_X_PATH, mmap_mode="r")
    y = np.load(TRAIN_Y_PATH)

    return X, y


def load_test_data():
    """Load the test images and one-hot encoded labels."""
    X = np.load(TEST_X_PATH, mmap_mode="r")
    y = np.load(TEST_Y_PATH)

    return X, y


def one_hot_to_labels(y):
    """Convert one-hot encoded labels to integer class labels."""
    return np.argmax(y, axis=1)


def create_train_validation_split(
    X,
    y,
    validation_size=0.1,
    random_state=42,
):
    """
    Create a reproducible train/validation split.

    Labels are converted from one-hot format to integer labels
    for stratification and then returned in their original format.
    """
    labels = one_hot_to_labels(y)

    indices = np.arange(len(X))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )

    return (
        X[train_indices],
        X[validation_indices],
        y[train_indices],
        y[validation_indices],
    )


if __name__ == "__main__":
    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()

    print("Training images:", X_train.shape)
    print("Training labels:", y_train.shape)

    print("Test images:", X_test.shape)
    print("Test labels:", y_test.shape)

    labels = one_hot_to_labels(y_train)

    print("Class distribution:")
    unique, counts = np.unique(labels, return_counts=True)

    for class_id, count in zip(unique, counts):
        print(f"Class {class_id}: {count}")