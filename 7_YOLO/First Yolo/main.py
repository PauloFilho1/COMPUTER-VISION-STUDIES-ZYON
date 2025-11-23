import cv2
from ultralytics import YOLO

# 1. Initial Configuration
video = cv2.VideoCapture('airport.mp4')

# Load YOLOv8 Nano model.
# Model hierarchy: n (nano), s (small), m (medium), l (large), x (extra large)
# Automatically downloads 'yolov8n.pt' on first run.
model = YOLO('yolov8n.pt')

# Print internal dictionary
print(model.names)

while True:
    check, img = video.read()
    if not check:
        break   

    # Original resizing maintained
    img = cv2.resize(img, (1090, 720))

    # 2. Inference + Tracking
    # persist=True: Enables tracking (maintains identity across frames)
    # classes=[0]: Filter COCO classes -> 0=Person
    # conf=0.2: Confidence threshold
    results = model.track(img, persist=True, classes=[0], conf=0.2)

    # 3. Visualization
    # The .plot() method automatically draws boxes, labels, and confidence scores
    plotted_img = results[0].plot()

    # Display on screen
    cv2.imshow('Video', plotted_img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()