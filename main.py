#import library
import numpy as np
import datetime
import glob
import os
import cv2
import gc

from keras.models import load_model
from pandas import DataFrame
from random import choice

#import local package
import dataset as ds
import siamese_train as st
import counting
import recognition

#save time when the programs start
start_time = datetime.datetime.now()

#import HaarCascade as Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#model properties
epoch = 10000
model_file = 'model.h5'
v_split = 0.3
dir_logs = 'log'
dir_logs_tb = 'log_tb'

#data set properties
width = 105
height = 105
channel = 3
dir_datatrain = 'datatrain'
dir_datatrain_new = 'datatrain_fc'
normalized = True
face_localization = True
datatrain_file = 'datatrain.npz'

#running task
run = 'none'

#counting properties
input_video = cv2.VideoCapture('video_test/output10.avi')
dir_data_captured = 'data'

#recognition properties
dir_data_test = 'datatest'
dir_data_test_batch = 'data_test/6'
dir_recognize = 'recognize'

label = 0
#result_table = []
result_filename = []
result_label = []

xls_file = 'result6.xlsx'

#FaceDetection function
def counting_from_video():
    
    print('Running counting process...')
    print('---------------------------\n')

    if not os.path.exists(dir_data_captured):
        os.makedirs(dir_data_captured)
        
    num = 0
    frame_number = 0
    face = []

    n_unique = 0

    flag_loss = 0
    empty = 1

    while(True):
        ret, frame = input_video.read()
        frame = cv2.resize(frame,(680,480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #he = cv2.equalizeHist(gray)
        
        x,y,w,h = -1,-1,-1,-1
        #detecting face in the frame
        faces = face_cascade.detectMultiScale(frame, 1.1, 5)

        for (x,y,w,h) in faces:
            #for every face found in the frame, there will be drawn a box
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)

            face = gray[y+1:y+h-1,x+1:x+w-1]
        
        if x!=-1 and y != -1 and frame_number % 10 == 0:
            face = cv2.resize(face, (105,105))
            cv2.imwrite('data/'+str(num)+'.jpg',face)
           
            filename = 'data/'+str(num)+'.jpg'

            n_unique += counting.uniqueCount(model_file, filename)
            print("Unique count: ", n_unique,'\n')

            num += 1
            
        frame_number += 1
    
        cv2.imshow('Frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

def recognition_data_test():
    
    print('Running recognition process...')
    print('---------------------------\n')

    if not os.path.exists(dir_recognize):
        print('Generating recognize data...')
        print('---------------------------\n')
    
        img_number = 1
    
        classes = glob.glob(os.path.join(dir_data_test,'*'))
        classes = [os.path.basename(w) for w in classes]
    
        for fields in classes:
            
            path = os.path.join(dir_data_test,fields, '*jpg')
            files = glob.glob(path)
        
            file_path = choice(files)

            image = cv2.imread(file_path)
        
            faces = face_cascade.detectMultiScale(image, 1.3, 5)

            for (x,y,w,h) in faces:
                image = image[y:y+h, x:x+w]
        
            image = cv2.resize(image, (width,height),0,0, cv2.INTER_LINEAR)
        
            if not os.path.exists(dir_recognize):
                os.makedirs(dir_recognize)

            print(img_number)
            if(img_number / 10 < 1):        
                cv2.imwrite(dir_recognize+'/00'+str(img_number)+'.jpg',image)
            else:
                cv2.imwrite(dir_recognize+'/0'+str(img_number)+'.jpg',image)

            print('({}) selected from ({})'.format(file_path,fields))
                
            img_number += 1
        
    print('\nCollecting extracted data features from '+dir_recognize)
    print('---------------------------\n')

    for filename in glob.glob(os.path.join(dir_recognize, '*.jpg')):
        recognition.createFaceDataExtract(model_file, filename, (width,height))
        print('Face data: ',filename);

    print('\nFace Recognition and labeling')
    print('---------------------------\n')
    
    path = os.path.join(dir_data_test_batch, '*.jpg')
    files = glob.glob(path)
        
    for filename in files:
        print('--------------------------------------')
        print('Labeling file:',filename,'\n')
        label = recognition.faceRecognition(model_file, filename, (width,height))

        print("\nImage: ",filename," labeled as ",label+1,"\n")
            

        #result = "{} - {}".format(filename, label+1)

        #result_table.append(result)

        result_filename.append(filename)
        result_label.append(label+1)

    df_result = DataFrame({'Filename': result_filename, 'Label': result_label})
    df_result.to_excel(xls_file, sheet_name='Sheet1', index=False)
        
#main function
def main():

    n_unique = 0
    
    #checking model
    print('[INFO] Searching for model...(',model_file,')\n')
    if os.path.isfile(model_file):
        print('[INFO] Model found.\n')

    else:
        print('[INFO] Model not found!\n')
        print('[INFO] Creating model...')
        print('---------------------\n')

    # Loading Dataset
        print('Load datatrain...')
        if os.path.isfile(datatrain_file):
            print('[INFO] Datatrain found!\n')
            	
        else:
            print('[INFO] Datatrain not found!\n')
            print('[INFO] Creating datatrain...\n')

            if(face_localization):
                if not os.path.exists(dir_datatrain_new):
                    ds.preprocess(dir_datatrain,dir_datatrain_new,(width,height))

                ds.generate_data(datatrain_file,dir_datatrain_new+'/'+dir_datatrain,(width,height),normalized)
                
            else:
                ds.generate_data(datatrain_file,dir_datatrain,(width,height),normalized)	

        print('[INFO] Start training...\n')
        if not os.path.exists(dir_logs):
            os.makedirs(dir_logs)

        if not os.path.exists(dir_logs_tb):
            os.makedirs(dir_logs_tb)
            os.makedirs(dir_logs_tb+'/training')
            os.makedirs(dir_logs_tb+'/validation')

        st.train(model_file,epoch,datatrain_file,width,height,channel,v_split)

    if(run == 'recognition'):
        recognition_data_test()
        
    elif(run == 'counting'):
        
        counting_from_video()

    gc.collect()
    
if __name__ == "__main__":
    main()
