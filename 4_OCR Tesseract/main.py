import cv2
import pytesseract as pt

img = cv2.imread('img01.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# If processing a book page with shadows, use AdaptiveThreshold here
# img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)

data = pt.pytesseract.image_to_data(img_gray)

# The 'data' variable is the large string resembling the table above
for x, line in enumerate(data.splitlines()):

    if x != 0:
        line = line.split()  # Converts string to list
        
        if len(line) == 12:
            # Checks if text is not empty
            if len(line[11]) > 0:
                x_pos, y_pos, w, h = int(line[6]), int(line[7]), int(line[8]), int(line[9])
                word = line[11]
                print(f"Detected word: {word}")
                
                cv2.rectangle(img, (x_pos, y_pos), (w+x_pos, h+y_pos), (0, 255, 0), 2)
                cv2.putText(img, word, (x_pos, y_pos-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

img = cv2.resize(img, None, fx=0.4, fy=0.4)
cv2.imshow('Result', img)
cv2.waitKey(0)