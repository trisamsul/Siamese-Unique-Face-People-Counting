from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import load_model

from IPython.display import display
from PIL import Image

from scipy.spatial.distance import euclidean

import numpy as np
import cv2

# variable to save extracted data
ext_data = []

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def faceRecognition(model_file, img_path, img_size):

    # distance value
    dist = 0.0
    
    # Load model
    base_model = load_model(model_file)

    # Extract features from an arbitrary intermediate layer
    # like the block4 pooling layer in VGG19
    model = Model(inputs = base_model.input, outputs = base_model.get_layer('Feature_Extract_1').output)

    # Load image test, convert into array, normalizing
    img = cv2.imread(img_path)
    
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:
        img = img[y:y+h, x:x+w]
        
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, img_size,0,0, cv2.INTER_LINEAR)

    x = image.img_to_array(img)
    x = np.multiply(x, 1.0 / 255.0)
    x = np.expand_dims(x, axis=0)

    # get the features 
    features = model.predict([x,x])

    print(features.shape)

    min_dist = np.sum((features - ext_data[0])**2) / 4096
    index_label = 0

    if ext_data:
        i=0
        while i < len(ext_data):

            dist = np.sum((features - ext_data[i])**2) /4096
            print('Distance value: ',dist)

            if dist < min_dist:
                min_dist = dist
                index_label = i
            
            i += 1        
    else:
        print('(Extracted feature data is empty)\n')
        
    return index_label
                    
def createFaceDataExtract(model_file, img_path, img_size):

    base_model = load_model(model_file)

    model = Model(inputs = base_model.input, outputs = base_model.get_layer('Feature_Extract_1').output)

    img = cv2.imread(img_path)
        
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, img_size,0,0, cv2.INTER_LINEAR)
    
    x = image.img_to_array(img)
    x = np.multiply(x, 1.0 / 255.0)
    x = np.expand_dims(x, axis=0)

    features = model.predict([x,x])

    ext_data.append(features)
