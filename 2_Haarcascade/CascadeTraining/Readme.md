# Custom Haar Cascade Object Detection

This project provides the necessary tools to capture samples and test a custom Haar Cascade Classifier. The goal of this repository is to guide you through creating your own object detection model from scratch.

⚠️ **Note:** This repository does **not** include a pre-trained `cascade.xml` file. You must capture your own images and train the model yourself to generate it.

## 📂 Project Structure

    CascadeTraining/
    ├── Photos/
    │   ├── n/              # Negative Images (Backgrounds - Add manually)
    │   └── p/              # Positive Images (Captured via script)
    ├── capture.py          # Script to capture positive samples from webcam
    ├── test.py             # Script to test the model (requires cascade.xml)

## 🛠️ Dependencies

    pip install opencv-python

## 🚀 Workflow

### 1. Data Collection (Positives)
Use `capture.py` to collect images of the object you want to detect.

1. Run the script:

    python capture.py

2. Position the object in front of the camera.
3. Press **'q'** on your keyboard to capture and save a sample.
   * The image will be automatically converted to grayscale, resized to 220x220, and saved in `Photos/p/`.
   * **Recommendation:** Capture at least 50-100 positive images.

### 2. Data Collection (Negatives)
You must manually populate the `Photos/n/` folder.
* **What to add:** Images that do **not** contain the object you want to detect (e.g., empty walls, landscapes, random objects).
* **Recommendation:** You need considerably more negative images than positives (e.g., if you have 100 positives, aim for 200-300 negatives).

### 3. Training the Model
Since this project focuses on the pipeline, you need to use a training tool to generate the classifier.

1. Use a tool like **Cascade Trainer GUI** (Windows) or OpenCV's `opencv_traincascade` (Linux/CLI).
2. Point the training tool to your `Photos/p` (positives) and `Photos/n` (negatives) folders.
3. Start the training process.
4. **Output:** The training will generate a file named `cascade.xml`.
5. **Action:** Move this `cascade.xml` file to the root folder of this project (alongside `test.py`).

### 4. Testing
**Prerequisite:** Ensure you have placed the generated `cascade.xml` in the project root.

1. Run the test script:

    python test.py

2. The webcam will open, and the system will draw green rectangles around detected objects in real-time.

## ⚙️ Troubleshooting
* **"Error: Empty dataset"**: Ensure you added images to `Photos/n`.
* **"File not found"**: If `test.py` fails, double-check that your trained model is named exactly `cascade.xml` and is in the same folder as the script.

---
**Author:** Paulo Ferreira de Castro Filho (Zyon)