import cv2

# Applying Haar cascade

video_capture = cv2.VideoCapture('Files/people.mp4')
classifier = cv2.CascadeClassifier('Files/cascades/haarcascade_fullbody.xml')

while True:
    check, frame = video_capture.read()
    if not check:
        print("End of video or read error.")
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Stores a list of rectangles (x, y, width, height) for each detected body
    objects = classifier.detectMultiScale(gray_frame, minSize=(50, 50), scaleFactor=1.1)

    for x, y, w, h in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Frame', frame)
    cv2.waitKey(10)
