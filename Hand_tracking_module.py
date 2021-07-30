import cv2
import mediapipe as mp
import time

class handdetector():
    def __init__(self,mode=False,maxhands=2,detection_conf=0.5,track_conf=0.5):
        self.mode = mode
        self.maxhands= maxhands
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxhands,self.detection_conf,self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for currhand in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, currhand, self.mpHands.HAND_CONNECTIONS)

        return img

    def findpos(self,img, handnumber=0,draw=True):

        landmark_list = []
        if self.results.multi_hand_landmarks:
            currhand = self.results.multi_hand_landmarks[handnumber]

            for id, lm in enumerate(currhand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                landmark_list.append([id,cx,cy])
                # if draw and (id==8 or id==4):
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return landmark_list



def main():
    pTime = 0
    ctime = 0

    cap = cv2.VideoCapture(0)
    detector = handdetector()
    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        landmark_list = detector.findpos(img)
        if len(landmark_list)!=0:
            print(landmark_list[8])


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2,(255,255,0),2)

        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()