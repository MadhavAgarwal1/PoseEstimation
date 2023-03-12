import cv2
import mediapipe as mp 
import time

# mp_drawing_styles = mp.solutions.drawing_styles     
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()        #creating pose

cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read()

    # this img is in BGR and mediapipe uses RGB therefore...we convert it into RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)

    if results.pose_landmarks: 
        # Draw pose landmarks on the image. (0 to 32 points are there)
    
         # mpDraw.draw_landmarks(img, results.pose_landmarks)  # it make all the points
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)  # it connect all the points
        # mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())  # it make points color different


        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape     #height, width, no. of channels(here is 3(which are Red, Blue, Green))
            # print("img.shape",img.shape)
            # print(id, lm)          #lm-> landmark values are the ratio of images

            # to get actual pixel values of lm
            cx, cy = int(lm.x*w) , int(lm.y*h)

            #drawing circle at particular points
            cv2.circle(img, (cx, cy), 3, (255,0,0), cv2.FILLED)
        
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # put text in the frame 
    cv2.putText(img, str(int(fps)), (40,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  

    cv2.imshow("Image Name", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

