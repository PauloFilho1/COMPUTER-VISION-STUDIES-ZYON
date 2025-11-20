# Real-Time Emotion & Gesture Recognition CNN

This project implements a complete Deep Learning pipeline for Computer Vision, focused on real-time emotion and gesture recognition. The system covers everything from data collection (via webcam) to training a Convolutional Neural Network (CNN) and live inference.

## 📋 Overview

The project consists of three main modules:
1. **Capture:** Script to build your own dataset using the webcam.
2. **Training:** Image preprocessing and CNN training using TensorFlow/Keras.
3. **Inference:** Real-time video feed classification using the trained model.

## 🛠️ Technologies

* **Python 3.x**
* **OpenCV:** Video capture and image processing.
* **TensorFlow / Keras:** Neural network construction and training.
* **NumPy:** Array and matrix manipulation.
* **Matplotlib:** Performance visualization (Loss/Accuracy graphs).
* **Scikit-learn:** Data splitting (train/test).

## 📦 Installation

Run the command below to install the necessary dependencies:

    pip install opencv-python numpy matplotlib scikit-learn tensorflow

## 🚀 How to Run

The project is structured as follows:
* `capture.py` (Capture code)
* `CNN Training.py` (CNN training code)
* `inference.py` (Inference code)

### 1. Data Collection (`capture.py`)

⚠️ **IMPORTANT:** This repository does **not** include a pre-recorded dataset. You must generate your own dataset using this script before attempting to train the model.

**Recommendation:** For optimal model performance, capture **at least 700 images per class**.

1. Create a folder named `Images` in the project root.
2. Inside `Images`, create numeric subfolders for each class (e.g., `0`, `1`, `2`).
3. In the code, check if the save path points to the correct folder (e.g., `Images/0/`).
4. Run the script and press **'s'** to save samples.

### 2. Training (`CNN Training.py`)

This script loads the images, applies preprocessing (grayscale, equalization, normalization), and trains the network.

1. Ensure the `Images` folder is populated with your captured data.
2. Run the script (use quotes due to the space in the filename):

    python "CNN Training.py"

3. The script will generate performance graphs and save the final model as `gesture_model.h5`.

**Training Parameters:**
* **Epochs:** 15
* **Batch Size:** 32
* **Input Shape:** 128x128x1 (Grayscale)

### 3. Real-Time Inference (`inference.py`)

This script loads the `gesture_model.h5` model and classifies webcam images.

1. Run the script:

    python inference.py

2. The system will display the detected class and the confidence percentage on the screen.

**Class Legend (default in code):**
* **0:** Happy
* **1:** Angry
* **2:** Happy + gesture

## 🧠 Model Architecture

The CNN was designed to be lightweight and efficient:
* **Input:** 128x128 pixels.
* **Convolutional Layers:** 3 blocks with progressive filters (60 -> 30 -> 30), ReLU activation, and MaxPooling.
* **Regularization:** Dropout (0.5) to prevent overfitting.
* **Classification:** Dense layer with 500 nodes followed by Softmax.

---
**Author:** Paulo Ferreira de Castro Filho (Zyon)