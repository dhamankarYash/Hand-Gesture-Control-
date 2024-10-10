import cv2
import time
import mediapipe as mp
import math
import numpy as np


class handDetector():
    def __init__(self,mode=False,maxHands=2,modelcomplexity=1,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.modelcomplexity=modelcomplexity
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.modelcomplexity,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils

    def findHands(self,img,draw=True):    
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLMS,self.mpHands.HAND_CONNECTIONS)

        return img  

    def findPosition(self,img,handNo=0,draw=True):
        lmList=[]
        
        if self.results.multi_hand_landmarks:
            myhands=self.results.multi_hand_landmarks[handNo]
            
            for i_d, lm in enumerate(myhands.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([i_d,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),10,(190,120,260),cv2.FILLED)
        return lmList


cap=cv2.VideoCapture(0)
cap.set(3,1280)#width of the cap
cap.set(4,720)#height of the cap

detector=handDetector(detectionCon=0.7)

iTime=0
length=50
while True:
    success, img= cap.read()
    img=cv2.flip(img,1)
    
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)>0:
        x1,y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2
        
        cv2.circle(img,(x1,y1),9,(190,120,260),cv2.FILLED)
        cv2.circle(img,(x2,y2),9,(190,120,260),cv2.FILLED)
        cv2.line(img, (x1,y1),(x2,y2),(190,120,260),3)
        cv2.circle(img,(cx,cy),9,(190,120,260),cv2.FILLED)

        length=math.hypot(x2-x1,y2-y1)

        vol=np.interp(length,[50,300],[-65.25,0])
        
        if length<50:
            cv2.circle(img,(cx,cy),9,(0,255,0),cv2.FILLED)
        elif length>175:
            cv2.circle(img,(cx,cy),9,(0,165,310),cv2.FILLED)
        else:
            cv2.circle(img,(cx,cy),9,(0,255,255),cv2.FILLED)

    volBar=np.interp(length,[50,300],[400,150])
    volPer=int(np.interp(length,[50,300],[0,100]))
    cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,255),cv2.FILLED)
    cv2.rectangle(img,(50,150),(85,400),(0,220,160),3)
    cv2.putText(img,f"Vol: {volPer}%",(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,220,160),1)

    cTime=time.time()
    fps=int(1/(cTime-iTime))
    iTime=cTime

    cv2.putText(img,f"fps: {fps}",(40,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,220,160),2)
    
    cv2.imshow("Image",img)
    #cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
