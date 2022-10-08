import cv2


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    # Cut down frame to 250*250 pixels
    # plt.imshow(frame[120:120+250,200:200+250,:])
    frame = frame[120:120+250,200:200+250,:]

    # Collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # create unique file path
        imgname = os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))
        # write out anchor image
        cv2.imwrite(imgname,frame)

    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # create unique file path
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # write out positive image
        cv2.imwrite(imgname, frame)

    cv2.imshow('Image Collection', frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()