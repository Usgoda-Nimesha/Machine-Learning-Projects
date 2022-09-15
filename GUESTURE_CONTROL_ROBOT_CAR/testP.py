import cv2
import mediapipe as mp
import time
import numpy as np

import Hand_Tracking_Module as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(1)
detector = htm.handDetector()
while True:
    success, img = cap.read()
    blank_image = np.zeros((480, 640, 3), np.uint8)
    img = detector.findHands(img, blank_image)
    lmlist = detector.findPosition(blank_image, draw=False)
    if len(lmlist) != 0:
        print(lmlist[4])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(blank_image, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
