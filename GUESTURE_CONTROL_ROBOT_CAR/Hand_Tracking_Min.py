import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    blank_image = np.zeros((480, 640, 3), np.uint8)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw the lines connecting the points
            mpDraw.draw_landmarks(blank_image, handLms, mpHands.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = blank_image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                if id == 0:
                    cv2.circle(blank_image, (cx,cy), 10,(255,0,255),cv2.FILLED)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(blank_image, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", blank_image)
    cv2.waitKey(1)
