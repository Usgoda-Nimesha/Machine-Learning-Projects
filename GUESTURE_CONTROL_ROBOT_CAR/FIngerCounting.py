import cv2
import time
import os
import Hand_Tracking_Module as htm
import rccar as rc
import numpy as np

wCam, hCam = 640, 480
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
carController = rc.carController()


pTime = 0
fingerCountChanged = 0

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    blank_image = np.zeros((480, 640, 3), np.uint8)
    img = detector.findHands(img, blank_image)
    lmList = detector.findPosition(blank_image)
    # print(lmList)
    if len(lmList) != 0:
        fingers = []
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        currentFingerCount = fingers.count(1)
        #print(currentFingerCount)
        if currentFingerCount != fingerCountChanged and currentFingerCount==5:
            carController.forward()
        else:
            carController.stop()
        fingerCountChanged = currentFingerCount

    # h, w, c = overlayList[0].shape
    # img[0:h, 0:w] = overlayList[0]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (480, 40), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
