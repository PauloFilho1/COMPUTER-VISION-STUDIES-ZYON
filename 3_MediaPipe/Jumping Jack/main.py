import cv2
import mediapipe as mp
import math

pose_module = mp.solutions.pose
pose_detector = pose_module.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

drawing_utils = mp.solutions.drawing_utils
counter = 0
state_flag = True
video_capture = cv2.VideoCapture('jumping jack.mp4')

while True:
    success, frame = video_capture.read()
    if not success:
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(frameRGB)
    points = results.pose_landmarks
    h, w, _ = frame.shape

    if points:
        drawing_utils.draw_landmarks(frame, points, pose_module.POSE_CONNECTIONS)

        right_foot_y = int(points.landmark[pose_module.PoseLandmark.RIGHT_FOOT_INDEX].y * h)
        right_foot_x = int(points.landmark[pose_module.PoseLandmark.RIGHT_FOOT_INDEX].x * w)
        left_foot_y = int(points.landmark[pose_module.PoseLandmark.LEFT_FOOT_INDEX].y * h)
        left_foot_x = int(points.landmark[pose_module.PoseLandmark.LEFT_FOOT_INDEX].x * w)

        left_hand_y = int(points.landmark[pose_module.PoseLandmark.LEFT_INDEX].y * h)
        left_hand_x = int(points.landmark[pose_module.PoseLandmark.LEFT_INDEX].x * w)
        right_hand_y = int(points.landmark[pose_module.PoseLandmark.RIGHT_INDEX].y * h)
        right_hand_x = int(points.landmark[pose_module.PoseLandmark.RIGHT_INDEX].x * w)

        cv2.circle(frame, (right_foot_x, right_foot_y), 15, (0, 255, 0), cv2.FILLED)  # right foot
        cv2.circle(frame, (left_foot_x, left_foot_y), 15, (0, 255, 0), cv2.FILLED)   # left foot
        cv2.circle(frame, (left_hand_x, left_hand_y), 15, (0, 255, 0), cv2.FILLED)   # left hand
        cv2.circle(frame, (right_hand_x, right_hand_y), 15, (0, 255, 0), cv2.FILLED)  # right hand

        hand_distance = math.hypot(right_hand_x - left_hand_x, right_hand_y - left_hand_y)  # reference distance <= 100
        foot_distance = math.hypot(left_foot_x - right_foot_x, left_foot_y - right_foot_y)  # reference distance <= 50

        print(f'Hands distance: {hand_distance} | Feet distance: {foot_distance}')

        if state_flag == True and hand_distance <= 150 and foot_distance >= 150:
            counter = counter + 1
            state_flag = False

        if hand_distance > 150 and foot_distance < 150:
            state_flag = True

        text = f'Count: {counter}'

        cv2.rectangle(frame, (20, 240), (320, 120), (255, 0, 0), -1)
        cv2.putText(frame, text, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        #print(counter)

    cv2.imshow("Results", frame)
    cv2.waitKey(1)
