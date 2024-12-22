# Aygaz-Goruntu-Isleme
https://www.kaggle.com/code/handeuygun/notebook12cc38dc12/edit

# CNN-Based Image Classification with Augmentation and Preprocessing

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for multi-class image classification. The dataset used includes animal images, and the model is trained to classify 10 selected classes. The project also includes preprocessing steps, data augmentation, and evaluation on manipulated test sets to assess robustness.

---

## Features

- **Dataset Preparation**: Filters and copies specific animal classes into a new directory structure for training and testing.
- **Preprocessing**: Images are resized to 128x128, converted to RGB, and normalized.
- **Data Augmentation**: Includes random rotations, translations, flips, and other transformations to enhance training robustness.
- **CNN Model Architecture**:
  - Three convolutional layers with increasing filter sizes.
  - Batch normalization and max-pooling layers for improved performance.
  - Fully connected layers with dropout for overfitting prevention.
- **Evaluation**: Assesses model accuracy on:
  1. Original test set. -> 63.08%
  2. Brightness and contrast-adjusted test set. 9.79%
  3. Gray World color-corrected test set. 9.79%
  4. Gray World normalized test set. 9.79%

---

## Requirements

- Python 3.7 or higher
- Libraries:
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `opencv-python`
  - `Pillow`
  - `scikit-learn`

Install the required libraries using:
```bash
pip install tensorflow numpy pandas opencv-python pillow scikit-learn
