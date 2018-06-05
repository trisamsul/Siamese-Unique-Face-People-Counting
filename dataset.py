import cv2
import os
import numpy as np
import glob
from random import randint
from random import choice
from time import sleep

def generate_data(dataset_file, root_path, image_size, normalized):

    pairs_1 = []
    pairs_2 = []
    labels = []

    pair_same = 0
    pair_dif = 0

    classes = glob.glob(os.path.join(root_path,'*'))
    classes = [os.path.basename(w) for w in classes]

    print('Load dataset...')
    for fields in classes:   
        index = classes.index(fields)
        print('Read directory: {} (index: {})'.format(fields, index))
        print('---------------------------------------------')
        path = os.path.join(root_path,fields, '*jpg')
        files = glob.glob(path)

        list_except = classes[:]
        list_except.remove(fields)

        for f1 in files:
            #IMAGE 1
            image_1 = cv2.imread(f1)

            #resize and convert into array
            image_1 = cv2.resize(image_1, image_size,0,0, cv2.INTER_LINEAR)
            #image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
            image_1 = image_1.astype(np.float32)
            
            if normalized:
                image_1 = np.multiply(image_1, 1.0 / 255.0)
               
            #image_1 = np.expand_dims(image_1, axis=2)

            #IMAGE 2 - same class as image 1
            i = files.index(f1)

            if(i != len(files) - 1):
                for j in range(i+1, len(files)):
                   
                    image_2 = cv2.imread(files[j])

                    #resize and convert into array
                    image_2 = cv2.resize(image_2, image_size,0,0, cv2.INTER_LINEAR)
                    #image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
                    image_2 = image_2.astype(np.float32)

                    if normalized:
                        image_2 = np.multiply(image_2, 1.0 / 255.0)
                    
                    #image_2 = np.expand_dims(image_2, axis=2)

                    print('Image 1: '+f1)
                    print('Image 2: '+files[j]+'\n')

                    pairs_1.append(image_1)
                    pairs_2.append(image_2)
                    labels.append(1.0)

                    pair_same += 1

            #IMAGE 3 - different class from image 1
            for x in range(10):
                f_rand_fields = str(choice(list_except))
                f3_fields_path = os.path.join(root_path,f_rand_fields, '*jpg')
                f3_files = glob.glob(f3_fields_path)
                f3_path = choice(f3_files)

                image_3 = cv2.imread(f3_path)

                #resize and convert into array
                image_3 = cv2.resize(image_3, image_size,0,0, cv2.INTER_LINEAR)
                #image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
                image_3 = image_3.astype(np.float32)

                if normalized:
                    image_3 = np.multiply(image_3, 1.0 / 255.0)
                
                #image_3 = np.expand_dims(image_3, axis=2)

                print('Image 1: '+f1)
                print('Image 3: '+f3_path+'\n')
                
                pairs_1.append(image_1)
                pairs_2.append(image_3)
                labels.append(0.0)

                pair_dif += 1
            
    pairs_1 = np.array(pairs_1)
    pairs_2 = np.array(pairs_2)
    labels = np.array(labels)

    print('Total: ',len(labels))
    print('Same Pair: ',pair_same)
    print('Different Pair: ',pair_dif)
    print('\n======================\n')
    
    np.savez(dataset_file,X1=pairs_1,X2=pairs_2,Y=labels)

def preprocess(root_path, root_path_new, image_size):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    if not os.path.exists(root_path_new):
        os.makedirs(root_path_new)

    classes = glob.glob(os.path.join(root_path,'*'))
    classes = [os.path.basename(w) for w in classes]

    print('Load dataset...')
    for fields in classes:   
        index = classes.index(fields)
        print('Read directory: {} (index: {})'.format(fields, index))
        print('---------------------------------------------')
        path = os.path.join(root_path,fields, '*jpg')
        files = glob.glob(path)

        for f1 in files:
            
            print('Image: '+f1)

            image = cv2.imread(f1)

            faces = face_cascade.detectMultiScale(image, 1.3, 5)

            for (x,y,w,h) in faces:
                image = image[y:y+h, x:x+w]

            image = cv2.resize(image, image_size,0,0, cv2.INTER_LINEAR)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #image = cv2.equalizeHist(image)

            newpath = os.path.join(root_path_new,root_path,fields)
            file_path = os.path.join(root_path_new,f1)

            if not os.path.exists(newpath):
                os.makedirs(newpath)

            cv2.imwrite(file_path,image)
