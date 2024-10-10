import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

iTime=0

while True:
    success,img=cap.read()
    
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for i_d, lm in enumerate(handLMS.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(i_d,cx,cy)
            
            mpDraw.draw_landmarks(img,handLMS,mpHands.HAND_CONNECTIONS)



    cTime=time.time()
    fps=int(1/(cTime-iTime))
    iTime=cTime

    cv2.putText(img,f"fps: {fps}",(40,70),cv2.FONT_HERSHEY_COMPLEX,1,(350,130,100),2)

    
    cv2.imshow("Image",img)
    #cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
