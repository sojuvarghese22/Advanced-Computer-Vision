import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
cap = cv2.VideoCapture(r'E:\NEW DATA\Practising folder\Advanced Computer Vision2\videos\first.mp4')
pTime = 0
while True:
    success, img = cap.read()
    stretch_near = cv2.resize(img, (500,800))
 
    imgRGB = cv2.cvtColor(stretch_near, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(stretch_near, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = stretch_near.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(stretch_near, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(stretch_near, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", stretch_near)
    
    cv2.waitKey(1)
