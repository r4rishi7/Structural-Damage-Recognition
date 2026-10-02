# Structural Damage Recognition

## Overview

Structural Damage Recognition is a Machine Learning project focused on the automatic classification of structural images into two categories:

- **Damaged**
- **Undamaged**

The project uses image-based machine learning techniques to investigate how automated systems can assist in identifying visible structural damage.

---

## Problem Statement

Manual inspection of structures can be time-consuming and may require significant human effort. An automated image-based classification system can assist in the initial identification of potentially damaged structures.

The objective of this project is to develop and evaluate a machine learning model capable of classifying structural images as **Damaged** or **Undamaged**.

---

## Objectives

- Explore a structural image dataset.
- Understand the distribution of damaged and undamaged samples.
- Preprocess the image data for machine learning.
- Train classification models for damage-state recognition.
- Evaluate model performance using appropriate metrics.
- Visualize and analyze the obtained results.

---

## Dataset

The project uses the **PEER Φ-Net — Task 2: Damage State** dataset.

The task is a binary image-classification problem:

| Class | Description |
|---|---|
| Damaged | Structural image showing visible damage |
| Undamaged | Structural image without visible damage |

The original dataset files are not stored in this repository because of their large size.

Dataset download and preparation details will be documented in the `data/` directory.

---

## Project Structure

```text
Structural-Damage-Recognition/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
├── src/
│   └── README.md
│
├── results/
│   └── README.md
│
├── docs/
│   └── README.md
│
├── .gitignore
├── README.md
└── requirements.txt