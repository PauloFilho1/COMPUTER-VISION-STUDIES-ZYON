import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)
# Hand detection configured for a single hand

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)

    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    points = []

    if handsPoints:
        for hand_landmarks in handsPoints:
            # Draw hand landmarks and connections
            mpDraw.draw_landmarks(img, hand_landmarks, hand.HAND_CONNECTIONS)

            for idx, coord in enumerate(hand_landmarks.landmark):
                cx, cy = int(coord.x * w), int(coord.y * h)
                cv2.putText(img, str(idx), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                points.append((cx, cy))

        fingertips = [8, 12, 16, 20]
        finger_count = 0

        # The image behaves like the fourth quadrant of the Cartesian plane, hence the inverted Y-axis comparison
        if hand_landmarks:
            # Thumb count
            if points[4][0] < points[2][0]:
                finger_count += 1

            # Other fingers count
            for tip in fingertips:
                if points[tip][1] < points[tip - 2][1]:
                    finger_count += 1

        cv2.rectangle(img, (80, 10), (200, 110), (255, 0, 0), -1)
        cv2.putText(img, str(finger_count), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
