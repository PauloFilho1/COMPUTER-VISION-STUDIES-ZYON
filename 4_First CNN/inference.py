import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ==========================================
# CONSTANTS AND SETTINGS
# ==========================================
MODEL_PATH = 'gesture_model.h5'
IMAGE_DIMENSIONS = (128, 128)   # Input size expected by the model
THRESHOLD = 0.60                # Confidence threshold for classification

# ==========================================
# MODEL LOADING
# ==========================================
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Fatal error: Failed to load '{MODEL_PATH}'. Details: {e}")
    exit()

# Video capture configuration
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

def preprocessing(img):
    """
    Applies preprocessing pipeline: Grayscale, Histogram Equalization, and Normalization.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

def getClassName(classNo):
    """
    Maps class index to the corresponding label.
    """
    if classNo == 0: 
        return 'Happy'
    elif classNo == 1: 
        return 'Angry'
    elif classNo == 2: 
        return 'Happy + gesture'
    return 'NOT IDENTIFIED'

print("Capture started.")

while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Error capturing frame.")
        break

    # =================================================
    # IMAGE PROCESSING
    # =================================================
    
    # Conversion to array and resizing
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1]))
    
    # Preprocessing filters
    img = preprocessing(img)
    
    # Processed input visualization (Debug View)
    cv2.imshow("Debug - Model Input", img)
    
    # Dimension adjustment to tensor format (Batch, Height, Width, Channels)
    img = img.reshape(1, IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1], 1)

    # =================================================
    # INFERENCE
    # =================================================
    predictions = model.predict(img, verbose=0)
    indexVal = np.argmax(predictions)
    probabilityValue = np.amax(predictions)

    # =================================================
    # RENDERING (UI)
    # =================================================
    if probabilityValue > THRESHOLD:
        # Render label and confidence score
        label = getClassName(indexVal)
        confidence = f"{round(probabilityValue * 100, 1)}%"
        
        cv2.putText(
            imgOriginal, label, (50, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            imgOriginal, confidence, (50, 90),
            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2
        )
    else:
        # Render low-confidence state
        cv2.putText(
            imgOriginal, "LOW CONFIDENCE", (50, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2
        )

    cv2.imshow("Real-time Classification", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
