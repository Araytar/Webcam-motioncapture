import mediapipe as mp
import cv2
import time

#Video Capture
cap = cv2.VideoCapture(0)

#Img Overlay Paiter init
mpdraw = mp.solutions.drawing_utils

#Tracker innit
mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpBody = mp.solutions.pose
body = mpBody.Pose()

mpFace = mp.solutions.face_mesh
face = mpFace.FaceMesh()

while True:
    #read and convert image
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #return lm's
    handsRes = hands.process(imgRGB)
    bodyRes = body.process(imgRGB)
    faceRes = face.process(imgRGB)

    #Draw Landmarks
    if handsRes.multi_hand_landmarks:
        for hand_landmarks in handsRes.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    if bodyRes.pose_landmarks:
        mpdraw.draw_landmarks(img, bodyRes.pose_landmarks, mpBody.POSE_CONNECTIONS)

    if faceRes.multi_face_landmarks:
        for faceLms in faceRes.multi_face_landmarks:
            mpdraw.draw_landmarks(img, faceLms, mpFace.FACEMESH_CONTOURS)

    #Show Image
    cv2.imshow("Tracker", img)

    #Break on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()