import cv2

video = cv2.VideoCapture(0)

sample = 1

while True:
    check, img = video.read()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f'Images/0/im{sample}.jpg', img)
        print(f'image saved {sample}')
        sample += 1

    cv2.imshow('Capture', img)
    cv2.waitKey(1)
