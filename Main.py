#import library
import numpy as np
import tensorflow as tf
import datetime
import glob
import os
import cv2

from keras.models import load_model

#import local package
import dataset as ds
import siamese_train as st
import counting

#text properties variables
font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText    = (10,20)
bottomLeftCornerOfText = (10,220)
fontScale              = 0.4
fontColor              = (255,255,255)
lineType               = 1

#save time when the programs start
start_time = datetime.datetime.now()

#import HaarCascade as Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#model properties
model_file = 'model.h5'

#data set properties
width = 105
height = 105
channel = 3
dataset_file = 'dataset_face.npz'
dir_dataset = 'dataset'
dir_data = 'data'
epoch = 100
normalized = True

#video input
input_video = cv2.VideoCapture('video_dataset/example-01.mp4')

#PreProcess function
def preprocess(frame):

    #converting frame image from RGB to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #filtering image using Histogram Equalization
    #he = cv2.equalizeHist(gray)

    #converting frame image from Grayscale to RGB so the frame can be stacked
    res = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return res

#FaceDetection function
def facedetection():

    num = 0
    frame_number = 0
    face = []

    n_visit = 0
    n_unique = 0

    flag_loss = 10

    while(True):
        ret, frame = input_video.read()
        frame = cv2.resize(frame,(680,480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        x,y,w,h = -1,-1,-1,-1
        #detecting face in the frame
        faces = face_cascade.detectMultiScale(frame, 1.1, 5)

        for (x,y,w,h) in faces:
            #for every face found in the frame, there will be drawn a box
            cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(255,0,0),1)

            face = gray[y+1:y+h-1,x+1:x+w-1]
        
        if x!=-1 and y != -1 and frame_number % 20 == 0:
            face = cv2.resize(face, (105,105))
            cv2.imwrite('data/'+str(num)+'.jpg',face)
           
            filename = 'data/'+str(num)+'.jpg'

            n_unique += counting.uniqueCount(model_file, filename, dir_data)
            print("Unique count: ", n_unique)

            num += 1

        """
        #Count Visit
        if x!=-1 and y != -1 and frame_number % 10 == 0:
            flag_loss = 10
            
        else:
            flag_loss -= 1

        if flag_loss <= 0:
            n_visit += 1
        """
    
        
        #print("Visit count: ", n_visit,'\n')
        
        frame_number += 1
    
        cv2.imshow('Frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
        
#main function
def main():

    #checking model
    print('Loading Model...\n')
    if os.path.isfile(model_file):
        print('Model found.')
    else:
        print('Model file is not exist!\n')
        print('Creating model...')
        print('---------------------\n')
        
        st.train(epoch,dataset_file,dir_dataset,width,height,channel,normalized)

    facedetection()
    
    """
    for filename in glob.glob(os.path.join(dir_data, '*.jpg')):
        print('File: ',filename)
        n_unique += counting.uniqueCount(model_file, filename, dir_data)
        print("Unique count: ", n_unique,'\n')
    """
    
    #Max Distance (for treshold)
    md = counting.glob_max_dist
    mdr = sum(md) / float(len(md))

    print('----------------')
    print('Average Max Different Value: ',mdr)
    
    
if __name__ == "__main__":
    main()
