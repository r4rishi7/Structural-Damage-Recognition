import numpy as np
import tensorflow as tf


# =========================
# Configuration
# =========================

MODEL_PATH = "models/inceptionv3.keras"

TEST_X_PATH = "DATA/task2_damage_state_1/task2_X_test.npy"
TEST_Y_PATH = "DATA/task2_damage_state_1/task2_y_test.npy"

BATCH_SIZE = 16


# =========================
# Main prediction function
# =========================

def main():

    print("Loading trained model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")
    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    print("\nLoading test data...")

    X_test = np.load(
        TEST_X_PATH,
        mmap_mode="r"
    )

    y_test = np.load(
        TEST_Y_PATH,
        mmap_mode="r"
    )

    print(f"Test images : {X_test.shape}")
    print(f"Test labels : {y_test.shape}")

    # Convert one-hot labels to class indices
    true_labels = np.argmax(y_test, axis=1)

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Convert softmax probabilities to predicted classes
    predicted_labels = np.argmax(predictions, axis=1)

    # =========================
    # Accuracy
    # =========================

    accuracy = np.mean(
        predicted_labels == true_labels
    )

    print("\n==============================")
    print("Prediction Results")
    print("==============================")
    print(f"Samples           : {len(true_labels)}")
    print(f"Correct           : {np.sum(predicted_labels == true_labels)}")
    print(f"Incorrect         : {np.sum(predicted_labels != true_labels)}")
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")
    print("==============================")

    # =========================
    # Display sample predictions
    # =========================

    print("\nFirst 10 predictions:")

    for i in range(min(10, len(predicted_labels))):

        print(
            f"Sample {i + 1:02d} | "
            f"True: {true_labels[i]} | "
            f"Predicted: {predicted_labels[i]} | "
            f"Probabilities: {predictions[i]}"
        )

    # =========================
    # Save predictions
    # =========================

    output_path = "results/inceptionv3_predictions.npz"

    np.savez(
        output_path,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=predictions
    )

    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    main()