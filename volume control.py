import cv2
import mediapipe as mp
import time
import Hand_tracking_module as htm
import math

#############
wcam, hcam = 800,800
#############

pTime = 0
ctime = 0

cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)

detector = htm.handdetector(detection_conf=0.7)
while True:
    success, img = cap.read()
    img = detector.findhands(img)
    landmark_list = detector.findpos(img,draw=True)
    if len(landmark_list) != 0:
        pass
        # print(landmark_list[4],landmark_list[8])

        x1,y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]
        cx,cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img,(x1,y1), 15, (0,255,255),cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0,255,255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (0,255,255), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(0,255,255),3)

        length = math.hypot(x2-x1,y2-y1)
        print(length)

        if length<=50:
            cv2.circle(img, (cx, cy), 15, (255,255,0), cv2.FILLED)

    # cv2.rectangle(img,(50,150),(85,400).(0,255,0),cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (255, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)