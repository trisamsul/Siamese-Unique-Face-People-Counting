from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import load_model

from scipy.spatial.distance import euclidean

import numpy as np
import cv2

# variable to save extracted data
ext_data = []
#glob_max_dist = []

def uniqueCount(model_file, img_path, dir_data):

    #status if same face found ('0' none, '1' found)
    status = 0

    # distance value
    dist = 0.0
    max_dist = 0.0
    
    # Load model
    base_model = load_model(model_file)

    # Extract features from an arbitrary intermediate layer
    # like the block4 pooling layer in VGG19
    model = Model(inputs = base_model.input, outputs = base_model.get_layer('Feature_Extract').output)

    img = image.load_img(img_path, target_size=(105, 105))
    x = image.img_to_array(img)
    x = np.multiply(x, 1.0 / 255.0)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)

    #print(np.max(x))
    
    # get the features 
    features = model.predict(x)

    if ext_data:
        i=0
        while i < len(ext_data) and status == 0:
            
            dist = np.sum((features - ext_data[i])**2)
            print('Distance value: ',dist)

            if dist < 350.0:
                status = 1

            if max_dist < dist:
                max_dist = dist
            
            i += 1        
    else:
        print('(Extracted feature data is empty)\n')
        
    ext_data.append(features)
    #glob_max_dist.append(max_dist)

    #print('Max Distance Value: ',max_dist)
    
    if status == 0:
        return 1
    else:
        return 0
                    
