from keras.models import Model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array

import numpy as np
import tensorflow as tf
import dataset as ds
import os

def train(epoch,dataset_file,dir_dataset,width,height,channel,normalized):

    # Loading Dataset
    print('Load Dataset...\n')
    if os.path.isfile(dataset_file):
        dsw = np.load(dataset_file)
        X = dsw['X']
        Ybin = dsw['Ybin']
        Y = dsw['Y']
    else:
        X, Y, Ybin, imgns, cls = ds.load_data(dir_dataset,(105,105),channel,normalized)
        np.savez(dataset_file,X=X, Y=Y, Ybin=Ybin, imgns=imgns, cls=cls)

    holdout_idx = ds.get_holdout_idx(Y,0.7,True)
    Xtrain = X[holdout_idx==1]
    Ytrain = Ybin[holdout_idx==1]
    #Xtest = X[holdout_idx==2]
    #Ytest = Ybin[holdout_idx==2]

    # Feature Extraction Layer
    inputs = Input(shape=(width, height, channel))

    conv_layer = ZeroPadding2D(padding=(2,2))(inputs) 
    conv_layer = Conv2D(64, (10, 10), activation='relu')(conv_layer) 
    conv_layer = MaxPooling2D()(conv_layer) 
    conv_layer = Conv2D(128, (7, 7), activation='relu')(conv_layer)
    conv_layer = MaxPooling2D()(conv_layer)
    conv_layer = Conv2D(128, (4, 4), activation='relu')(conv_layer) 
    conv_layer = MaxPooling2D()(conv_layer)
    conv_layer = Conv2D(256, (4, 4), activation='relu')(conv_layer) 
    conv_layer = MaxPooling2D(name='Feature_Extract')(conv_layer) 

    # Flatten feature map to Vector with 576 element.
    flatten = Flatten()(conv_layer) 

    # Fully Connected Layer
    outputs = Dense(11, activation='relu')(flatten)

    model = Model(inputs=inputs, outputs=outputs)

    # Adam Optimizer and Cross Entropy Loss
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    #model.fit(Xtrain,Ytrain,batch_size = 10 ,nb_epoch=100)
    model.fit(Xtrain, Ytrain, epochs=epoch, batch_size=None)

    # Print Model Summary
    #print(model.summary())

    model.save("model.h5")
