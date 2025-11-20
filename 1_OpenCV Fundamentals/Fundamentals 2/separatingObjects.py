import cv2
import os

# Load and resize source image
image = cv2.imread('objects.jpg')
image = cv2.resize(image, (600, 500))

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection (Canny)
canny_image = cv2.Canny(gray_image, 30, 200)

# Morphological closing (dilation followed by erosion)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed_image = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, kernel)

# Extract external contours only
contours, hierarchy = cv2.findContours(
    closed_image,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)

# Ensure output directory exists
output_dir = 'Cropped Images'
os.makedirs(output_dir, exist_ok=True)

object_index = 1
min_area = 1000

for contour in contours:
    # Compute the area of the current contour
    area = cv2.contourArea(contour)

    if area > min_area:
        # Axis-aligned bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        cropped_object = image[y:y + h, x:x + w]

        # Save cropped object
        output_path = os.path.join(output_dir, f'object{object_index}.jpg')
        cv2.imwrite(output_path, cropped_object)

        # Draw bounding box and contour on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.drawContours(image, contour, -1, (0, 255, 0), 2)

        object_index += 1

# Display results
cv2.imshow('Image', image)
cv2.imshow('Grayscale image', gray_image)
cv2.imshow('Canny edges', canny_image)
cv2.imshow('Morphological close', closed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
