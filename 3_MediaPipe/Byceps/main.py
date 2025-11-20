import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
import math
import time
from datetime import datetime
import openpyxl

video_capture = cv2.VideoCapture('vd01.mp4')

detector = PoseDetector()
# PoseDetector instance

counter = 0
control_flag = False
# Flag to ensure each repetition is counted only once
target_count = 4
start_time = 0
elapsed_time = 0
start_datetime = ''
end_datetime = ''

workbook = openpyxl.load_workbook('data.xlsx')
# Load workbook file
worksheet = workbook['data']
# Load worksheet
def save_data(start_datetime,end_datetime,elapsed_time,counter,target_count):
    last_row_A = len(worksheet['A'])
    # Number of filled rows in column A of worksheet 'worksheet'
    next_row = last_row_A+1
    record_id = last_row_A
    worksheet[f'A{next_row}'].value = record_id
    worksheet[f'B{next_row}'].value = start_datetime
    worksheet[f'C{next_row}'].value = end_datetime
    worksheet[f'D{next_row}'].value = elapsed_time
    worksheet[f'E{next_row}'].value = counter
    worksheet[f'F{next_row}'].value = target_count
    workbook.save('data.xlsx')
    print('Data saved successfully!')

while True:
    check,img = video_capture.read()
    if not check: break
    img = cv2.resize(img,(1280,720))
    detector.findPose(img,draw=False)
    landmark_list,_ = detector.findPosition(img,draw=False)

    if len(landmark_list)>=1:
        right_shoulder_x = landmark_list[12][0] # Point 12: right shoulder
        right_shoulder_y = landmark_list[12][1]

        right_elbow_x = landmark_list[14][0] # Point 14: right elbow
        right_elbow_y = landmark_list[14][1]

        right_wrist_x = landmark_list[16][0] # Point 16: right wrist
        right_wrist_y = landmark_list[16][1]

        cv2.circle(img,(right_shoulder_x,right_shoulder_y),15,(0,0,255),-1)
        cv2.circle(img, (right_elbow_x, right_elbow_y), 15, (0, 0, 255), -1)
        cv2.circle(img, (right_wrist_x, right_wrist_y), 15, (0, 0, 255), -1)

        cv2.circle(img,(right_shoulder_x,right_shoulder_y),10,(255,255,255),-1)
        cv2.circle(img, (right_elbow_x, right_elbow_y), 10, (255,255,255), -1)
        cv2.circle(img, (right_wrist_x, right_wrist_y), 10, (255,255,255), -1)

        cv2.line(img,(right_shoulder_x,right_shoulder_y),(right_elbow_x,right_elbow_y),(255,255,255),2)
        cv2.line(img, (right_elbow_x, right_elbow_y), (right_wrist_x, right_wrist_y), (255, 255, 255), 2)

        delta_x = abs(right_wrist_x-right_shoulder_x)
        delta_y = abs(right_wrist_y-right_shoulder_y)
        # abs = absolute value (always positive)
        tangent_arc = math.atan2(delta_y,delta_x)
        degrees_value = math.degrees(tangent_arc)
        final_value = int(degrees_value)

        cv2.putText(img,f'{final_value} degrees',(right_elbow_x+10,right_elbow_y),cv2.FONT_HERSHEY_COMPLEX,0.7,
                    (255,255,255),2)

        # Below 55 degrees
        if final_value <=55 and control_flag==False:
            counter +=1
            control_flag= True
        elif final_value >55:
            control_flag=False

        if counter==1 and start_time==0:
            start_time = time.time()
            # Current time
            start_datetime = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        elif counter>=1 and counter <target_count:
            elapsed_time = time.time()-start_time
        elif counter==target_count:
            end_datetime = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            save_data(start_datetime,end_datetime,elapsed_time,counter,target_count)
            start_time=0
            elapsed_time=0
            counter=0

        cvzone.putTextRect(img,f'Count: {counter}',(right_shoulder_x+10,right_shoulder_y+100),3.5,4,colorR=(0,0,255))
        cvzone.putTextRect(img, f'Time: {round(elapsed_time,2)}', (50,100), 4, 4, colorR=(0, 0, 255))

    cv2.imshow('Image',img)
    cv2.waitKey(10)
