# Structural Damage Recognition

## Overview

Structural Damage Recognition is a machine learning project for automatically classifying structural images into two categories:

- Damaged
- Undamaged

The project evaluates multiple machine learning and deep learning approaches using accuracy, precision, recall, F1-score, and confusion matrices.

## Objectives

- Prepare and analyze the structural damage dataset.
- Create reproducible training and validation splits.
- Develop a batch-based data loading pipeline.
- Evaluate multiple machine learning and deep learning models.
- Compare model performance.
- Reproduce the relevant paper-based experiments.
- Document the final results.

## Dataset

The dataset contains RGB structural images of size 224 x 224 x 3.

Dataset sizes:

| Dataset | Samples |
|---|---:|
| Training | 11,811 |
| Test | 1,460 |

The training data uses a stratified 90/10 split:

- Training: 10,629
- Validation: 1,182
- Test: 1,460

The test set is kept separate from training and validation.

### Class Distribution

Training:
- Undamaged: 6,282
- Damaged: 5,529

Test:
- Undamaged: 745
- Damaged: 715

### Preprocessing

Observed pixel statistics:

- Minimum: -123.68
- Maximum: 151.061
- Mean: 14.082709
- Standard deviation: 59.887745

The dataset is already provided in a preprocessed numerical format, so image values are not divided by 255.

## Models Evaluated

Six configurations were evaluated:

1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. RBF-SVM
4. MobileNetV1
5. InceptionV3
6. InceptionV3 Layer-288 + SVM

### Classical Machine Learning

KNN, Logistic Regression, and RBF-SVM use 2048-dimensional deep features extracted from the InceptionV3 global average pooling layer.

KNN:
- n_neighbors = 5
- weights = distance
- StandardScaler

Logistic Regression:
- max_iter = 2000
- random_state = 42
- StandardScaler

RBF-SVM:
- kernel = rbf
- C = 1.0
- gamma = scale
- StandardScaler

### InceptionV3

Training configuration:

- Batch size: 16
- Epochs: 10
- Validation split: 10%
- Random state: 42

The trained model is stored at:

`models/inceptionv3.keras`

Test results:

- Accuracy: 65.75%
- Precision: 69.30%
- Recall: 53.99%
- F1: 60.69%

### MobileNetV1

Test results:

- Accuracy: 76.78%
- Precision: 74.35%
- Recall: 80.28%
- F1: 77.20%

MobileNetV1 achieved the best overall performance.

### InceptionV3 Layer-288 + SVM

Layer 288 (`activation_90`) produces a 5 x 5 x 384 feature representation, flattened to 9600 dimensions.

SVM configuration:

- kernel = rbf
- C = 1.0
- gamma = 0.001

No gamma tuning was performed.

Test results:

- Accuracy: 59.59%
- Precision: 58.72%
- Recall: 58.88%
- F1: 58.80%

## Final Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| KNN | 61.44% | 62.75% | 52.31% | 57.06% |
| Logistic Regression | 64.32% | 64.92% | 59.02% | 61.83% |
| RBF-SVM | 65.96% | 68.41% | 56.64% | 61.97% |
| InceptionV3 | 65.75% | 69.30% | 53.99% | 60.69% |
| MobileNetV1 | **76.78%** | 74.35% | **80.28%** | **77.20%** |
| InceptionV3 Layer-288 + SVM | 59.59% | 58.72% | 58.88% | 58.80% |

## Confusion Matrices

### KNN

```text
[[523, 222],
 [341, 374]]