# Dataset

This folder contains the dataset used for the Structural Damage Recognition project.

## Dataset Source

The project uses the official PEER Phi-Net Task 2 - Damage State dataset for binary classification of structural images into:

- Damaged
- Undamaged

The dataset is not included in the GitHub repository because of its large size.

## Dataset Files

The dataset is organized into two folders:

### task2_damage_state_1/

Contains:

- task2_X_test.npy - test images
- task2_y_test.npy - test labels
- task2_y_train.npy - training labels
- license.txt
- README.txt

### task2_damage_state_2/

Contains:

- task2_X_train.npy - training images

## Dataset Dimensions

| Dataset | Shape |
|---|---:|
| Training images | (11811, 224, 224, 3) |
| Training labels | (11811, 2) |
| Test images | (1460, 224, 224, 3) |
| Test labels | (1460, 2) |

The images are RGB images with a spatial resolution of 224 x 224.

## Labels

The labels are stored in one-hot encoded format:

- [1, 0] - Class 0
- [0, 1] - Class 1

## Train/Validation/Test Split

The official test set is kept completely separate and is used only for final evaluation.

The training set is divided using a stratified 90/10 split:

- Training: 90% - 10,629 images
- Validation: 10% - 1,182 images
- Test: 1,460 images

The split uses random_state=42 to ensure reproducibility.

## Preprocessing

The dataset images are already numerically preprocessed in the supplied .npy files.

The project does not divide the pixel values by 255.

Observed pixel statistics are approximately:

- Minimum: -123.68
- Maximum: 151.06
- Mean: 14.08
- Standard deviation: 59.89

The data is loaded using NumPy memory mapping where appropriate to avoid unnecessarily loading the entire dataset into RAM.

## Download

The dataset files must be obtained separately from the official PEER Phi-Net Task 2 dataset source and placed in the directory structure described above.

After downloading, verify that the required .npy files are present before running the training or evaluation scripts.
