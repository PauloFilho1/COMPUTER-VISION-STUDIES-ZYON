# Computer Vision Studies 👁️

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Library](https://img.shields.io/badge/Library-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

## 📘 About

This repository serves as a living documentation of my journey into **Computer Vision** and **Deep Learning**. Here, I implement algorithms ranging from classical image processing techniques to modern Convolutional Neural Networks (CNNs).

The goal is to build a solid foundation for advanced research and a future Master's degree in the field.

## 📂 Project Directory

| Module / Project | Description | Tech Stack |
| :--- | :--- | :--- |
| **[1_OpenCV Fundamentals](./1_OpenCV%20Fundamentals)** | Basic image processing techniques: reading images, color spaces, thresholds, contours, and drawing functions. | `OpenCV` `NumPy` |
| **[2_Haarcascade](./2_Haarcascade)** | Object detection using Haar feature-based cascade classifiers (Face detection, Eye detection, and custom training). | `OpenCV` |
| **[3_MediaPipe](./3_MediaPipe)** | Real-time tracking solutions (Hand Tracking, Pose Estimation, Face Mesh) utilizing Google's MediaPipe framework. | `MediaPipe` `OpenCV` |
| **[4_OCR Tesseract](./4_OCR%20Tesseract)** | Optical Character Recognition (OCR) pipeline to extract and digitize text from images using the Pytesseract wrapper. | `pytesseract` `OpenCV` |
| **[5_Facial Recognition](./5_Facial%20Recognition)** | Face identification and verification system using the DeepFace framework (wrapping VGG-Face/FaceNet models). | `DeepFace` `OpenCV` |
| **[6_First CNN](./6_First%20CNN)** | **[Featured Project]** Real-time emotion/gesture classification pipeline using a custom CNN trained on a self-captured dataset. | `TensorFlow` `Keras` |
| **[7_YOLO](./7_YOLO)** | State-of-the-art real-time object detection implementation using the YOLOv8 architecture. | `PyTorch` `YOLOv8` |

## 🛠️ Technologies & Tools

The core stack used throughout these studies includes:

* **Languages:** Python 3.x
* **Core Libraries:** OpenCV, NumPy, Matplotlib, MediaPipe
* **Deep Learning:** TensorFlow/Keras and PyTorch
* **Environment:** Linux (Pop!_OS)

## 🚀 How to Use This Repository

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PauloFilho1/COMPUTER-VISION-STUDIES-ZYON.git
    ```

2.  **Explore the Folders:**
    * **Complex Projects (e.g., `4_First CNN`):** These folders contain their own `README.md` with specific instructions for training and inference.
    * **Study Folders (e.g., `1_OpenCV Fundamentals`):** These contain standalone Python scripts. You can simply run them to see the concepts in action.

3.  **Run a specific script:**
    ```bash
    cd "1_OpenCV Fundamentals"
    StartingOpenCV.py
    ```

## map Study Roadmap

My planned learning path for future updates:

- [x] Basic Image Processing (Filtering, Edges, Contours)
- [x] Object Detection with Classical Methods (Haar Cascades)
- [x] Landmark Detection (MediaPipe Hands/Pose)
- [x] CNNs for Classification (Custom Architectures)
- [ ] Transfer Learning (VGG16, ResNet, MobileNet)
- [x] Object Detection with Deep Learning (YOLO, SSD)
- [ ] Semantic Segmentation (U-Net)
- [ ] Generative Adversarial Networks (GANs)

---
**Author:** Paulo Ferreira de Castro Filho (Zyon)