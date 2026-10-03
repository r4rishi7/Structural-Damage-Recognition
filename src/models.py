import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model


def build_inceptionv3(input_shape=(224, 224, 3), num_classes=2):
    """
    Build an ImageNet-pretrained InceptionV3 model
    for binary structural-damage classification.
    """

    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze the pretrained backbone initially
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation="softmax")(x)

    model = Model(
        inputs=base_model.input,
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model