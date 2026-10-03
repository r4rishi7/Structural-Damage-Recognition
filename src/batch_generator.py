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


class DamageDataGenerator(Sequence):
    """
    Memory-efficient batch generator for the structural damage dataset.

    Images remain memory-mapped on disk and are loaded only when a batch
    is requested.
    """

    def __init__(
        self,
        indices,
        batch_size=32,
        shuffle=True,
    ):
        self.indices = np.asarray(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.X = np.load(
            TRAIN_X_PATH,
            mmap_mode="r",
        )

        self.y = np.load(TRAIN_Y_PATH)

        self.current_indices = self.indices.copy()

        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.current_indices) / self.batch_size))

    def __getitem__(self, index):
        """Return one batch."""
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
        """Shuffle indices after every epoch."""
        if self.shuffle:
            np.random.shuffle(self.current_indices)


if __name__ == "__main__":
    from data_loader import (
        load_training_arrays,
        get_train_validation_indices,
    )

    X_train, y_train = load_training_arrays()

    train_indices, validation_indices = (
        get_train_validation_indices(y_train)
    )

    generator = DamageDataGenerator(
        train_indices,
        batch_size=32,
        shuffle=True,
    )

    print("Number of batches:", len(generator))

    X_batch, y_batch = generator[0]

    print("Batch images:", X_batch.shape)
    print("Batch labels:", y_batch.shape)
    print("Image dtype:", X_batch.dtype)
    print("Label dtype:", y_batch.dtype)