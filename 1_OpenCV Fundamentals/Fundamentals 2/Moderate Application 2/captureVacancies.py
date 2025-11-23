import cv2
import pickle

img = cv2.imread('parking_lot.png')

parking_spaces = []

for x in range(69):
    spot = cv2.selectROI('parking_spaces', img, False)
    cv2.destroyWindow('parking_spaces')
    parking_spaces.append((spot))

    for x, y, w, h in parking_spaces:
        # y+h defines the bottom-right corner due to the downward coordinate system
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

with open('parking_spaces.pkl', 'wb') as file:
    pickle.dump(parking_spaces, file)