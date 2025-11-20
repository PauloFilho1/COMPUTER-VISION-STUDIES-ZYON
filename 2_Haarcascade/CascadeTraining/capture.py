import cv2

# Capturing samples for Haar Cascade training

video = cv2.VideoCapture(0)

sample_index = 1
while True:
    ret, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imshow('Capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        resized_sample = cv2.resize(gray_frame, (220, 220))
        cv2.imwrite(f'Photos/p/im{sample_index}.jpg', resized_sample)
        sample_index += 1

    cv2.imshow('Capture', frame)
    cv2.waitKey(1)
