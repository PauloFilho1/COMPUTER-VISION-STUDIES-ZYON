import cv2

# Thresholding (binarization)


'''image = cv2.imread('book_with_shadow.jpg')

image = cv2.resize(image, (700, 800))
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Global threshold: pixels above 127 are set to 255 (white), others to 0 (black)
_, th1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Adaptive Gaussian threshold: weighted mean of neighboring pixels
th2 = cv2.adaptiveThreshold(
    gray_image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    13,
    8
)

# Adaptive mean threshold: simple mean of neighboring pixels
th3 = cv2.adaptiveThreshold(
    gray_image,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    13,
    8
)

# In both adaptive methods above, the constant 8 is subtracted from the local mean

cv2.imshow('Original', image)
cv2.imshow('Grayscale', gray_image)
cv2.imshow('Threshold - Global', th1)
cv2.imshow('Threshold - Adaptive Gaussian', th2)
cv2.imshow('Threshold - Adaptive Mean', th3)

cv2.waitKey(0)'''


# Drawing shapes and text (applies to images and video frames)

'''video = cv2.VideoCapture('../Fundamentals 1/runners.mp4')

while True:
    check, frame = video.read()
    if not check:
        break

    frame = cv2.resize(frame, (700, 400))

    # Rectangle
    cv2.rectangle(frame, (50, 50), (100, 100), (255, 0, 0), 5)

    # Circle
    cv2.circle(frame, (150, 75), 30, (0, 0, 255), 5)

    # Line
    cv2.line(frame, (50, 130), (180, 130), (255, 255, 0), 2)

    # Text overlay
    text = "Egyptian Pyramids"
    cv2.putText(
        frame,
        text,
        (50, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow('Frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()'''


# Morphological operations

'''image = cv2.imread('pyramid.jpg')
image = cv2.resize(image, (500, 400))

# Convert original (color) image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur to reduce noise before edge detection
blur_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

# Edge detection using Canny
canny_image = cv2.Canny(image, 50, 100)

# Structuring element for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Dilation: thickens bright regions; useful for connecting broken edges
dilated_image = cv2.dilate(canny_image, kernel, iterations=5)

# Erosion: thins bright regions; useful for reducing small objects or noise
eroded_image = cv2.erode(canny_image, kernel, iterations=2)

# Opening (erosion -> dilation): effective for removing small bright noise
opening_image = cv2.morphologyEx(canny_image, cv2.MORPH_OPEN, kernel)

# Closing (dilation -> erosion): effective for closing small dark holes inside objects
closing_image = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original image', image)
cv2.imshow('Grayscale image', gray_image)
cv2.imshow('Blurred image', blur_image)
cv2.imshow('Canny edges', canny_image)
cv2.imshow('Dilated image', dilated_image)
cv2.imshow('Eroded image', eroded_image)
cv2.imshow('Opening', opening_image)
cv2.imshow('Closing', closing_image)

cv2.waitKey(0)
cv2.destroyAllWindows()'''
