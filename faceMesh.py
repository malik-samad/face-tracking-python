import cv2
import mediapipe as mp
import time

# capture video from camera[0] (means first camera)
cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh

faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

# Color and other specifications for points and connections of faceMesh
pointsDrawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0,250,0))
connectionDrawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(250,0,0))

showFaceMesh = True
while True:
    success, img = cap.read()

    # convert image from BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # process RGB image for face meshes
    results = faceMesh.process(imgRGB)

    # wait and get key if pressed on each iteration
    waitKey = cv2.waitKey(1)
    # if 'q' is pressed then exit this app
    if waitKey == ord('q'):
        break
    # toggle show face mesh option by pressing 'f' key on keyboard
    elif waitKey == ord('f'):
        showFaceMesh = not showFaceMesh

    # display 'show face mesh' and its value as text on the image
    cv2.putText(img, f"Show Face Mesh: {showFaceMesh}", (20, 90), cv2.FONT_ITALIC, 0.5, (100, 250, 0), 1)

    if results.multi_face_landmarks and showFaceMesh:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, facelms, mpFaceMesh.FACE_CONNECTIONS, pointsDrawSpec, connectionDrawSpec)

    # calculate frame per sec
    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime

    # display 'FPS:' and fps value as text on the image
    cv2.putText(img, f"FPS: {fps}", (20,50), cv2.FONT_ITALIC, 1, (100,250, 0), 1)

    # create window to show image on it, also set the title of the window as first arg
    cv2.imshow("Output window (Face recognitions)", img)
