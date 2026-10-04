import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3

from src.preprocessing import load_training_data, load_test_data
from src.preprocessing import create_train_validation_split


OUTPUT_DIR = "results/classical_ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
X, y_onehot = load_training_data()
X_test, y_test_onehot = load_test_data()

print("Creating stratified train/validation split...")
X_train, X_val, y_train, y_val = create_train_validation_split(
    X,
    y_onehot,
    validation_size=0.1,
    random_state=42
)

y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)
y_test = np.argmax(y_test_onehot, axis=1)

print("Building InceptionV3...")
base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

layer_288 = base_model.layers[288]

print("Layer 288:", layer_288.name)
print("Layer 288 output:", layer_288.output.shape)

feature_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=layer_288.output
)


def extract_features(X_data, name):
    print(f"\nExtracting {name} features...")
    features = feature_model.predict(
        X_data,
        batch_size=32,
        verbose=1
    )

    print(f"{name} raw shape:", features.shape)

    features = features.reshape(features.shape[0], -1)

    print(f"{name} flattened shape:", features.shape)

    return features.astype(np.float32)


X_train_features = extract_features(X_train, "train")
X_val_features = extract_features(X_val, "validation")
X_test_features = extract_features(X_test, "test")


output_path = os.path.join(
    OUTPUT_DIR,
    "inceptionv3_layer288_features.npz"
)

np.savez_compressed(
    output_path,
    X_train=X_train_features,
    y_train=y_train,
    X_val=X_val_features,
    y_val=y_val,
    X_test=X_test_features,
    y_test=y_test
)

print("\nSaved:", output_path)
print("Train:", X_train_features.shape)
print("Validation:", X_val_features.shape)
print("Test:", X_test_features.shape)