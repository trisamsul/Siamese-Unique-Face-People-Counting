import keras.backend as K
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Lambda
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from time import sleep

def train(model_file,epoch,dataset_file,width,height,channel,v_split):
    
    #Load Datatrain
    dsw = np.load(dataset_file)
    X1 = dsw['X1']
    X2 = dsw['X2']
    Y = dsw['Y']
    
    # Feature Extraction Layer
    inputs_1 = Input(shape=(width, height, channel))
    inputs_2 = Input(shape=(width, height, channel))

    encoded_input_1 = convolutional_network(inputs_1,'Feature_Extract_1')
    encoded_input_2 = convolutional_network(inputs_2,'Feature_Extract_2')

    l1_distance_layer = Lambda(
        lambda tensors: K.abs(tensors[0] - tensors[1]))

    l1_distance = l1_distance_layer([encoded_input_1, encoded_input_2])

    prediction = Dense(units=1, activation='sigmoid')(l1_distance)

    model = Model(inputs=[inputs_1,inputs_2], outputs=prediction)

    # Adam Optimizer and Cross Entropy Loss
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    filepath="log/checkpoint-best-weight.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=10)
    #tensorboard = TensorBoard(log_dir='./log_tb', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    
    #model.fit(Xtrain,Ytrain,batch_size = 10 ,nb_epoch=100)
    history = model.fit(x=[X1,X2], y=Y, validation_split=v_split, epochs=epoch, batch_size=32, callbacks=[checkpoint,TrainValTensorBoard(write_graph=False)])
    
    # Plotting Training History
    print('\n=====================================')
    print(model.summary())
    print('\n=====================================')
    print(history.history.keys())
    print('=====================================')
    
    fig1 = plt.gcf()
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.draw()
    fig1.savefig('graph/acc_'+str(v_split)+'.png', dpi=100)
    

    fig2 = plt.gcf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.draw()
    fig2.savefig('graph/loss_'+str(v_split)+'.png', dpi=100)

    #save model
    model.save(model_file)

def convolutional_network(inputs, name):

    conv_layer = ZeroPadding2D(padding=(2,2))(inputs) 
    conv_layer = Conv2D(64, (10, 10), activation='relu')(conv_layer) 
    conv_layer = MaxPooling2D()(conv_layer) 
    conv_layer = Conv2D(128, (7, 7), activation='relu')(conv_layer)
    conv_layer = MaxPooling2D()(conv_layer)
    conv_layer = Conv2D(128, (4, 4), activation='relu')(conv_layer) 
    conv_layer = MaxPooling2D()(conv_layer)
    conv_layer = Conv2D(256, (4, 4), activation='relu')(conv_layer) 
    conv_layer = MaxPooling2D()(conv_layer) 

    # Flatten feature map to Vector with 576 element.
    flatten = Flatten()(conv_layer)

    # Fully Connected Layer
    connected = Dense(units=4096, activation='sigmoid',name=name)(flatten)
    
    return connected

def build_model_from_weight(weight_file, model_file, dataset_file, width, height, channel):
    
    dsw = np.load(dataset_file)
    X1 = dsw['X1']
    X2 = dsw['X2']
    Y = dsw['Y']
    
    # Feature Extraction Layer
    inputs_1 = Input(shape=(width, height, channel))
    inputs_2 = Input(shape=(width, height, channel))

    encoded_input_1 = convolutional_network(inputs_1,'Feature_Extract_1')
    encoded_input_2 = convolutional_network(inputs_2,'Feature_Extract_2')

    l1_distance_layer = Lambda(
        lambda tensors: K.abs(tensors[0] - tensors[1]))

    l1_distance = l1_distance_layer([encoded_input_1, encoded_input_2])

    prediction = Dense(units=1, activation='sigmoid')(l1_distance)

    model = Model(inputs=[inputs_1,inputs_2], outputs=prediction)

    # Adam Optimizer and Cross Entropy Loss
    adam = Adam(lr=0.0001)
    
    #load Weight
    model.load_weights(weight_file)
    
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(x=[X1,X2], y=Y, validation_split=0.3, epochs=1, batch_size=32)
    
    model.save(model_file)
    

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='log_tb/', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
