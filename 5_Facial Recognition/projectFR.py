import cv2
import os
import numpy as np
from deepface import DeepFace

# CPU Configuration (Lightweight and Fast)
MODEL_NAME = "ArcFace"      # Highly accurate and lightweight model
DETECTOR = "opencv"         # Fastest detector for CPU (less accurate than mediapipe, but very fast)
METRIC = "cosine"           # Comparison metric
THRESHOLD = 0.5             # Similarity threshold (lower is stricter)

database = {} # Dictionary {name: embedding}

def load_database():
    path = 'People'
    if not os.path.exists(path):
        print(f"Create the folder '{path}' and place photos there.")
        return

    print("Loading database...")
    files = os.listdir(path)
    
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(file)[0]
            img_path = os.path.join(path, file)
            
            try:
                # Generates numerical vector (embedding) of the saved image
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False
                )[0]["embedding"]
                
                database[name] = embedding
                print(f"Loaded: {name}")
            except Exception as e:
                print(f"Error loading {file}: {e}")

def verify_frame(frame, x, y, w, h):
    # Crops only the face to send to AI (saves processing)
    face_img = frame[y:y+h, x:x+w]
    
    try:
        # Generates embedding of the webcam face
        result = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend="skip" # Already detected with OpenCV below
        )
        current_embedding = result[0]["embedding"]

        # Compares with everyone in the database
        best_match = "Unknown"
        min_distance = float('inf')

        for name, saved_embedding in database.items():
            # Calculates Cosine Distance
            a = np.array(current_embedding)
            b = np.array(saved_embedding)
            distance = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            
            if distance < min_distance:
                min_distance = distance
                if distance < THRESHOLD:
                    best_match = name

        return best_match, min_distance

    except:
        return "Error", 1.0

def main():
    load_database()
    
    # Initializes webcam and lightweight OpenCV detector
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Converts to gray for detection (faster)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detects faces (OpenCV Haar Cascade is ultra lightweight)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        # Takes only the first face (usually largest area) as requested
        if len(faces) > 0:
            (x, y, w, h) = faces[0] 
            
            # Draws rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Calls verification (simple)
            name, dist = verify_frame(frame, x, y, w, h)
            
            # Writes the name
            label = f"{name} ({dist:.2f})"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow('Webcam - DeepFace Lite', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()