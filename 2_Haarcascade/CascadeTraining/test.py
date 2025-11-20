import cv2

# Load trained Haar cascade classifier (generated with Cascade Trainer GUI)
classifier = cv2.CascadeClassifier('cascade.xml')

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect objects in the grayscale frame
    objects = classifier.detectMultiScale(gray_frame, scaleFactor=1.1)

    if len(objects) == 0:
        pass
    if len(objects) != 0:
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Camera', frame)
    cv2.waitKey(1)
