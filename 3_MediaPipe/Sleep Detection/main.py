import cv2
import mediapipe as mp
import math
import time

video_capture = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

start_time = 0
status_code = ""

while True:
    check,img = video_capture.read()
    img = cv2.resize(img,(1000,720))
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)
    h,w,_ = img.shape

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            if face == None:
                break
            # Denormalize landmark coordinates

            right_eye_top_x,right_eye_top_y = int((face.landmark[159].x)*w),int((face.landmark[159].y)*h)
            right_eye_bottom_x, right_eye_bottom_y = int((face.landmark[145].x) * w), int((face.landmark[145].y) * h)
            left_eye_top_x,left_eye_top_y = int((face.landmark[386].x)*w),int((face.landmark[386].y)*h)
            left_eye_bottom_x, left_eye_bottom_y = int((face.landmark[374].x) * w), int((face.landmark[374].y) * h)

            cv2.circle(img,(right_eye_top_x,right_eye_top_y),1,(255,0,0),2)
            cv2.circle(img, (right_eye_bottom_x, right_eye_bottom_y), 1,(255, 0, 0), 2)
            cv2.circle(img, (left_eye_top_x, left_eye_top_y), 1,(255, 0, 0), 2)
            cv2.circle(img, (left_eye_bottom_x, left_eye_bottom_y), 1,(255, 0, 0), 2)

            right_eye_distance = math.hypot(right_eye_top_x-right_eye_bottom_x,right_eye_top_y-right_eye_bottom_y)
            left_eye_distance = math.hypot(left_eye_top_x - left_eye_bottom_x, left_eye_top_y - left_eye_bottom_y)

            if right_eye_distance <=10 and left_eye_distance <=10:
                cv2.rectangle(img,(100,30),(390,80),(0,0,255),-1)
                cv2.putText(img,"EYES CLOSED",(105,65),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
                state = "F"
                if state!=status_code:
                    start_time = time.time()
            else:
                cv2.rectangle(img,(100,30),(370,80),(0,255,0),-1)
                cv2.putText(img,"EYES OPEN",(105,65),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
                state = "A"
                start_time = time.time()
                duration = int(time.time()-start_time)

            if state == 'F':
                duration = int(time.time()-start_time)

            status_code = state

            if duration >=2:
                cv2.rectangle(img,(300,150),(850,220),(0,0,255),-1)
                cv2.putText(img,f'SLEEPING {duration} SEC',(310,200),cv2.FONT_HERSHEY_SIMPLEX,1.7,(255,255,255),5)
            print(duration)


    cv2.imshow('IMG',img)
    cv2.waitKey(1)
