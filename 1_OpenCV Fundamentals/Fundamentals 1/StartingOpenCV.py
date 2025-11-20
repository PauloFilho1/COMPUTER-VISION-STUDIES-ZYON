import cv2

# Image cropping and file saving

'''img = cv2.imread('pelicans.png')
dim = cv2.selectROI("Select cropping region", img, False)
print(dim)
cv2.destroyWindow('Select cropping region')

x = int(dim[0])
y = int(dim[1])
w = int(dim[2])
h = int(dim[3])

# Rows span from y to y + h (height) and columns from x to x + w (width), producing the selected crop.
image_crop = img[y:y+h, x:x+w]
path = 'cutouts/'
file_name = input('Enter output file name: ')

cv2.imwrite(f'{path}{file_name}.png', image_crop)  # Write cropped image to file
print('Image saved.')

cv2.waitKey(0)'''


# Webcam capture

'''camera = cv2.VideoCapture(0)
camera.set(3, 640)   # Frame width (pixels)
camera.set(4, 420)   # Frame height (pixels)
camera.set(10, 100)  # Brightness

while True:
    check, img = camera.read()

    cv2.imshow('WebCam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break'''



# Video playback from file

'''video = cv2.VideoCapture('runners.mp4')

while True:
    check, img = video.read()
    print(check)
    resized_img = cv2.resize(img, (640, 420))  # Resize frame to fixed resolution
    cv2.imshow('video', resized_img)
    cv2.waitKey(20)'''


# Static image loading and basic inspection

'''img = cv2.imread('pelicans.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

print(img)         # Raw pixel data (NumPy array)
print(img.shape)   # (height, width, number_of_channels)
cv2.imshow('Image', img)
cv2.imshow('Grayscale image', gray_img)
cv2.waitKey(0)'''
