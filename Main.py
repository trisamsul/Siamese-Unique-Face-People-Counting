#import library
import numpy as np
import cv2
import datetime

import dataset as ds

#text properties variables
font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText    = (10,20)
bottomLeftCornerOfText = (10,220)
fontScale              = 0.4
fontColor              = (255,255,255)
lineType               = 1

start_time = datetime.datetime.now()

#import HaarCascade as Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#PreProcess function
def PreProcess(frame):

    #converting frame image from RGB to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #filtering image using Histogram Equalization
    #he = cv2.equalizeHist(gray)

    #converting frame image from Grayscale to RGB so the frame can be stacked
    res = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return res

#FaceDetection function
def FaceDetection(frame,num):

    x,y,w,h = -1,-1,-1,-1
    #detecting face in the frame
    faces = face_cascade.detectMultiScale(frame, 1.1, 5)

    for (x,y,w,h) in faces:
        #for every face found in the frame, there will be drawn a box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        face = frame[y+1:y+h-1,x+1:x+w-1]

    #save captured face
    if x != -1 and y != -1 and num % 20 == 0:
        face = cv2.resize(face, (105,105))
        cv2.imwrite('data/'+str(num)+'.jpg',face)
    
    return frame

#Counting function
def Counting(count):

    #iteration for every new face that found
    count = count + 1
    
    return count

#main function
def main():

    #loading video source
    src = cv2.VideoCapture('video_dataset/output10.avi')
    #src = cv2.VideoCapture(0)
     
    #define total number of people
    count = 0
    num = 1

    while(True):
        #reading the video
        ret, ori = src.read()

        #resizing the frame size
        ori = cv2.resize(ori,(360,240))
        
        F1 = ori                                    #Frame 1 containing original video
        F2 = PreProcess(ori)                        #Frame 2 containing video result after doing Pre Process
        F3 = FaceDetection(PreProcess(ori),num)     #Frame 3 containing video result after doing Face Detection
        F4 = np.zeros((360,240,3), np.uint8)        #Frame 4 containing the final result
        F4 = cv2.resize(F4,(360,240))

        #collecting counting result
        count = Counting(count)

        #Putting text on the each frame
        cv2.putText(F1,'Source Video', 
            topLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.putText(F2,'Pre Processing', 
            topLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.putText(F3,'Face Detection', 
            topLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.putText(F4,'Since: '+str(start_time), 
            topLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.putText(F4,'Count: '+str(count), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        #stacking the frame
        StackF1F2 = np.hstack((F1,F2))
        StackF3F4 = np.hstack((F3,F4))

        #MainFrame containing all the frame for each process
        MainFrame = np.vstack((StackF1F2,StackF3F4))

        #showing the frame
        cv2.imshow('Main',MainFrame)

        #break whenever the user pressing the X button
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

        num = num + 1

    #release the video source
    src.release()
    
    #close all windows
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
