from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import load_model

from scipy.spatial.distance import euclidean

import numpy as np
import cv2

ext_data = 0

# define the CNN network
# Here we are using 19 layer CNN -VGG19 and initialising it
# with pretrained imagenet weights
base_model = load_model("model.h5")

# Extract features from an arbitrary intermediate layer
# like the block4 pooling layer in VGG19
model = Model(inputs=base_model.input, outputs=base_model.get_layer('Feature_Extract').output)

# load both image and preprocess it
img_path_1 = 'datatest/34.jpg'
img_path_2 = 'datatest/50.jpg'

img_1 = image.load_img(img_path_1, target_size=(105, 105))
x = image.img_to_array(img_1)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

img_2 = image.load_img(img_path_2, target_size=(105, 105))
y = image.img_to_array(img_2)
y = np.expand_dims(y, axis=0)
y = preprocess_input(y)

# get the features 
features_1 = model.predict(x)
features_2 = model.predict(y)

ext_data = []
ext_data.append(features_1)
ext_data.append(features_2)

i=0
while i < len(ext_data):
    print(ext_data[i])
    i+=1

dist = np.sum((features_2-features_1)**2) / 1.0e+06 
print(img_path_1, img_path_2)
print('Difference value: ',dist)
