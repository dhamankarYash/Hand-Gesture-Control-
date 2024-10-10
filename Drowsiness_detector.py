import cv2 as cv
import numpy as np
import mediapipe as mp
from pygame import mixer
import pygame

def open_len(arr):
    y_arr = []

    for _,y in arr:
        y_arr.append(y)

    min_y = min(y_arr)
    max_y = max(y_arr)

    return max_y - min_y

class faceDetector():
    def __init__(self, mode=False, maxFaces=1, refinelandmarks=True, modelcomplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode=mode
        self.maxFaces=maxFaces
        self.refinelandmarks=refinelandmarks
        self.modelcomplexity=modelcomplexity
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        print(type(self.mode),type(self.maxFaces),type(self.refinelandmarks),type(self.modelcomplexity),type(self.detectionCon),type(self.trackCon))
        
        self.mpFaces=mp.solutions.face_mesh
        self.face_mesh=self.mpFaces.FaceMesh(self.mode,self.maxFaces,self.refinelandmarks,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils
        
    def findFaces(self,img,draw=False):    
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.face_mesh.process(imgRGB)
        
        if self.results.multi_face_landmarks:
            for faceLMS in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLMS,self.mpFaces.FACE_CONNECTIONS)
    
        return img
    
    def findPosition(self,img,faceNo=0):
        facelmList=[]
        
        if self.results.multi_face_landmarks:
            myfaces=self.results.multi_face_landmarks[faceNo]
            facelmList=myfaces.landmark
        return facelmList

mixer.init()
pygame.init()
sound = mixer.Sound("Chord.wav")

iconimg = pygame.image.load('oie_zQZ30lQXiANK.png')
pygame.display.set_icon(iconimg)

RIGHT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

drowsy_frames = 0
max_left = 0
max_right = 0
width=1280
height=720

cap = cv.VideoCapture(0)
cap.set(3, width)           
cap.set(4, height)

screen=pygame.display.set_mode((width,height))
pygame.display.set_caption('Project Exhibition 2')
clock=pygame.time.Clock()

face_mesh = faceDetector()
while True:
    success, img = cap.read()

    img = cv.flip(img, 1)
    img_h, img_w = img.shape[:2]


    img = face_mesh.findFaces(img)
    facelmList=face_mesh.findPosition(img)

    if len(facelmList)>0:

        all_landmarks = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in facelmList])

        right_eye = all_landmarks[RIGHT_EYE]
        left_eye = all_landmarks[LEFT_EYE]

        cv.rectangle(img, (all_landmarks[247][0], all_landmarks[247][1]), (all_landmarks[233][0], all_landmarks[233][1]),(0, 255, 0), 0)
        cv.rectangle(img, (all_landmarks[453][0], all_landmarks[453][1]), (all_landmarks[467][0], all_landmarks[467][1]),(0, 255, 0), 1)

        len_left = open_len(right_eye)
        len_right = open_len(left_eye)

        if len_left > max_left:
            max_left = len_left

        if len_right > max_right:
            max_right = len_right

        cv.putText(img=img, text='Max: ' + str(max_left)  + ' Left Eye: ' + str(len_left), fontFace=0, org=(10, 30), fontScale=0.5, color=(0, 255, 0))
        cv.putText(img=img, text='Max: ' + str(max_right)  + ' Right Eye: ' + str(len_right), fontFace=0, org=(10, 50), fontScale=0.5, color=(0, 255, 0))

        if (len_left <= int(max_left / 2) + 1 and len_right <= int(max_right / 2) + 1):
            drowsy_frames += 1
        else:
            drowsy_frames = 0

        if (drowsy_frames > 20):
            cv.putText(img=img, text='ALERT', fontFace=0, org=(200, 300), fontScale=3, color=(0, 255, 0), thickness = 3)
            sound.play()

    img = cv.flip(img, 1)     
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    imgRGB=np.rot90(imgRGB)
    
    frame=pygame.surfarray.make_surface(imgRGB).convert()
    screen.blit(frame,(0,0))
    
    pygame.display.update()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            cap.release()
            cv.destroyAllWindows()
            exit()
            break
