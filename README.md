# DeepLearning

## Project: Détection précoce des pathologies sur les images médicales : Pneumonie vs Normal

This project focuses on the early detection of pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). The model is trained and validated on a public dataset of chest radiographs and is deployed via a Flask API with a user-friendly Streamlit interface for image upload and prediction.

---

## Features

- Binary classification of chest X-ray images into Normal or Pneumonia
- Data preprocessing with image resizing, normalization, and augmentation
- CNN architecture with convolutional, batch normalization, dropout, and dense layers
- Use of regularization techniques and early stopping to prevent overfitting
- Evaluation using confusion matrix, loss curves, and ROC AUC metric
- Deployment via Flask REST API
- Streamlit web application for easy interaction and visualization

---

## Installation and Setup

### Prerequisites

- Python 3.7+
- TensorFlow
- Flask
- Streamlit

### Steps

1. Clone this repository:

   ```bash
   git clone https://github.com/HajarAlouani/DeepLearning.git
   
