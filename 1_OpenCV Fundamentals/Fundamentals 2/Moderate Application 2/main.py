import cv2
import pickle
import numpy as np

parking_spaces = []

with open('parking_spaces.pkl', 'rb') as file:
    parking_spaces = pickle.load(file)

video = cv2.VideoCapture('video.mp4')

while True:
    check, img = video.read()
    if check == False:
        break
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve visibility
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    
    # Remove noise
    img_median = cv2.medianBlur(img_thresh, 5)
    
    # 3x3 matrix of ones
    kernel = np.ones((3, 3), np.int8)
    
    # Expand white regions and reduce black
    img_dilated = cv2.dilate(img_median, kernel)
    
    open_spots = 0

    for x, y, w, h in parking_spaces:
        spot_roi = img_dilated[y:y+h, x:x+w]
        count = cv2.countNonZero(spot_roi)
        cv2.putText(img, str(count), (x, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if count < 850:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            open_spots += 1
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.rectangle(img, (90, 0), (415, 60), (0, 255, 0), -1)
        cv2.putText(img, f'FREE: {open_spots}/69', (95, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

    cv2.imshow('video', img)
    cv2.imshow('video_thresh', img_dilated)
    cv2.waitKey(10)