from pathlib import Path

import numpy as np
from tensorflow.keras.utils import Sequence


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


class DamageDataGenerator(Sequence):
    """
    Memory-efficient generator for NumPy image datasets.

    Images are memory-mapped and loaded only batch-by-batch.
    """

    def __init__(
        self,
        X,
        y,
        indices=None,
        batch_size=32,
        shuffle=False,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

        if indices is None:
            self.indices = np.arange(len(y))
        else:
            self.indices = np.asarray(indices)

        self.current_indices = self.indices.copy()

        self.on_epoch_end()

    def __len__(self):
        """Return the number of batches."""
        return int(
            np.ceil(
                len(self.current_indices) / self.batch_size
            )
        )

    def __getitem__(self, index):
        """Return one batch of images and labels."""
        start = index * self.batch_size

        end = min(
            start + self.batch_size,
            len(self.current_indices),
        )

        batch_indices = self.current_indices[start:end]

        X_batch = np.asarray(
            self.X[batch_indices],
            dtype=np.float32,
        )

        y_batch = np.asarray(
            self.y[batch_indices],
            dtype=np.float32,
        )

        return X_batch, y_batch

    def on_epoch_end(self):
        """Shuffle training indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.current_indices)


def create_generators(
    validation_size=0.1,
    batch_size=32,
    random_state=42,
):
    """
    Create training, validation, and test generators.
    """

    from sklearn.model_selection import train_test_split

    X_train = np.load(
        TRAIN_X_PATH,
        mmap_mode="r",
    )

    y_train = np.load(
        TRAIN_Y_PATH,
    )

    X_test = np.load(
        TEST_X_PATH,
        mmap_mode="r",
    )

    y_test = np.load(
        TEST_Y_PATH,
    )

    # Convert one-hot labels to integer labels
    # only for stratified splitting.
    labels = np.argmax(y_train, axis=1)

    all_indices = np.arange(len(y_train))

    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )

    train_generator = DamageDataGenerator(
        X_train,
        y_train,
        indices=train_indices,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_generator = DamageDataGenerator(
        X_train,
        y_train,
        indices=validation_indices,
        batch_size=batch_size,
        shuffle=False,
    )

    test_indices = np.arange(len(y_test))

    test_generator = DamageDataGenerator(
        X_test,
        y_test,
        indices=test_indices,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_generator,
        validation_generator,
        test_generator,
    )


if __name__ == "__main__":

    (
        train_generator,
        validation_generator,
        test_generator,
    ) = create_generators()

    print("Train batches:", len(train_generator))
    print("Validation batches:", len(validation_generator))
    print("Test batches:", len(test_generator))

    X_train_batch, y_train_batch = train_generator[0]

    X_val_batch, y_val_batch = validation_generator[0]

    X_test_batch, y_test_batch = test_generator[0]

    print("\nTraining batch:")
    print("Images:", X_train_batch.shape)
    print("Labels:", y_train_batch.shape)

    print("\nValidation batch:")
    print("Images:", X_val_batch.shape)
    print("Labels:", y_val_batch.shape)

    print("\nTest batch:")
    print("Images:", X_test_batch.shape)
    print("Labels:", y_test_batch.shape)