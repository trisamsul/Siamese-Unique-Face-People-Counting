from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import load_model

from scipy.spatial.distance import euclidean

import numpy as np
import cv2

# variable to save extracted data
ext_data = []
glob_max_dist = []

def uniqueCount(model_file, img_path):

    #status if same face found ('0' none, '1' found)
    status = 0

    # Load model
    base_model = load_model(model_file)

    # Extract features from an arbitrary intermediate layer
    model = Model(inputs = base_model.input, outputs = base_model.get_layer('Feature_Extract_1').output)

    # Load image test, convert into array, normalizing
    img = image.load_img(img_path, target_size=(105, 105))
    x = image.img_to_array(img)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = x.astype(np.float32)
    x = np.multiply(x, 1.0 / 255.0)
    
    x = np.expand_dims(x, axis=2)
    
    # get the features 
    features = model.predict([x,x])

    if ext_data:
        i=0
        while i < len(ext_data):
            
            dist = np.sum((features - ext_data[i])**2) / 4096
            print('Distance value: ',dist)

            if dist < 0.003:
                status = 1
            
            i += 1        
    else:
        print('(Extracted feature data is empty)\n')
        
    ext_data.append(features)

    if status == 0:
        return 1
    else:
        return 0
