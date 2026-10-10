import numpy as np

# Caffe-style ImageNet channel means (BGR order).
IMAGENET_BGR_MEANS = np.array(
    [103.939, 116.779, 123.680],
    dtype=np.float32,
)


def restore_rgb_images(images):
    """
    Convert assumed Caffe-preprocessed BGR images
    back to RGB images in the 0-255 range.

    Intended for EfficientNetB0 preprocessing.
    """
    images = np.asarray(images, dtype=np.float32)

    # Undo ImageNet channel-mean subtraction.
    bgr_images = images + IMAGENET_BGR_MEANS

    # Convert BGR to RGB.
    rgb_images = bgr_images[..., ::-1]

    return np.ascontiguousarray(rgb_images)