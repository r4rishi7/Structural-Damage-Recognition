from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


HISTORY_PATH = Path("results/training/history.csv")
OUTPUT_DIR = Path("results/training")


def main():
    history = pd.read_csv(HISTORY_PATH)

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["accuracy"], label="Training Accuracy")
    plt.plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("InceptionV3 Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    accuracy_path = OUTPUT_DIR / "accuracy_curve.png"
    plt.savefig(accuracy_path, dpi=300)
    plt.show()
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["loss"], label="Training Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("InceptionV3 Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_path = OUTPUT_DIR / "loss_curve.png"
    plt.savefig(loss_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {accuracy_path}")
    print(f"Saved: {loss_path}")


if __name__ == "__main__":
    main()