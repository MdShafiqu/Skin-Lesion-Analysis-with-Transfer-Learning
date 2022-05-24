import cv2
#import Tkinter as tk
#import tkFileDialog
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import  Flatten, Dense, Activation,Convolution2D,MaxPooling2D, Dropout, BatchNormalization
from sklearn.cross_validation import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
#from image_Preprocessing import load_images,resize_images
import random
from keras.preprocessing.image import ImageDataGenerator
from   keras.callbacks import ModelCheckpoint


image_width  = 299
image_height = 299
image_size = (image_width, image_height)

def load_images(folder):
        image_list=[]
        for filename in os.listdir(folder):
            img=cv2.imread(os.path.join(folder,filename))
            if img is not None:
                image_list.append(img)
        return image_list
		
def resize_images(images):
		resize_image=[]
		imag=cv2.resize(image,(512,512))
		for image in images:
			resize_image.append(imag)
		return resize_image
		

X_Train=load_images('C:/Users/mislam/Final Project/data/Training Images')
X_Train=np.array(X_Train)
print(X_Train.shape)

X_Valid=load_images('C:/Users/mislam/Final Project/data/Validation Images')
X_Valid=np.array(X_Valid)
print(X_Valid.shape)

X_Test=load_images('C:/Users/mislam/Final Project/data/Testing Images')
X_Test=np.array(X_Test)
print(X_Test.shape)


data_Train=pd.read_csv("C:/Users/mislam/Final Project/data/ISIC-2017_Training_Part3_GroundTruth.csv")
data_Test=pd.read_csv("C:/Users/mislam/Final Project/data/ISIC-2017_Test_v2_Part3_GroundTruth.csv")
data_Valid=pd.read_csv("C:/Users/mislam/Final Project/data/ISIC-2017_Validation_Part3_GroundTruth.csv")
data_Train=data_Train.iloc[0:2000,1]
data_Test=data_Test.iloc[0:600,1]
data_Valid=data_Valid.iloc[0:150,1]
print (data_Train.shape)
print (data_Test.shape)
print (data_Valid.shape)


y_Train = LabelEncoder().fit_transform(data_Train)
y_Train = np_utils.to_categorical(y_Train)
y_Test = LabelEncoder().fit_transform(data_Test)
y_Test = np_utils.to_categorical(y_Test)
y_Valid = LabelEncoder().fit_transform(data_Valid)
y_Valid = np_utils.to_categorical(y_Valid)

Optimizer=Adam(lr=0.001)
objective='binary_crossentropy'
def center_normalize(x):
    return (x-K.mean(x))/K.std(x)

#Main CNN Architecture
baseMapNum = 32
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(64, 64,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.4))
	
# model=Sequential()
# #input layer
# model.add(Activation(activation=center_normalize, input_shape=(64, 64,3)))
# # convolutional layer
# model.add(Convolution2D(32,5,5,border_mode='same',activation='relu',dim_ordering='tf'))
# #pooling layer
# model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='tf'))
# # convolutional layer
# model.add(Convolution2D(64,3,3,border_mode='same',activation='relu',dim_ordering='tf'))
# # pooling layer
# model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='tf'))
model.add(Flatten())
# Relu 
model.add(Dense(128, activation='relu'))
model.add(Dense(y_Train.shape[1]))
model.add(Activation('sigmoid'))
print(model.summary())
#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(X_Train)
batch_size = 16
epochs=20

model.compile(loss=objective,optimizer=Optimizer,metrics=['accuracy'])
best_model = ModelCheckpoint("cnn_best.h5", monitor='val_loss', verbose=0, save_best_only=True)
model.fit(X_Train,y_Train,batch_size=16,nb_epoch=15,verbose=1,validation_data=(X_Valid, y_Valid), shuffle=True)
#model.fit_generator(datagen.flow(X_Train, y_Train, batch_size=batch_size),steps_per_epoch=X_Train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(X_Valid,y_Valid),callbacks=[best_model])
#Sigmoid Fully connected layer

from sklearn.metrics import roc_auc_score

predict = model.predict(X_Test, batch_size=1)
predict=predict[:,1]
#print(predict)

proba = model.predict_proba(X_Test, batch_size=1)
proba=proba[:,1]
#print(proba)

classes = model.predict_classes(X_Test, batch_size=1)
#print(classes)

#print(data_Test)

auc=roc_auc_score(data_Test, predict)
print ("AUC Score on Test Image: ",auc)

scores = model.evaluate(X_Test, y_Test)
#print(scores)
print ("Accuracy on Test Image: ", scores[1])

