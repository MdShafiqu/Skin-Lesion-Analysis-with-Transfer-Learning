from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
import random
import warnings

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.random.seed(0)
from   tqdm import *
from   sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from   keras           import backend as K
from   keras.models    import Model
from   keras.layers    import Dense, Input, BatchNormalization, Activation, merge, Dropout
from   keras.layers    import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D
from   keras.callbacks import ModelCheckpoint
from   keras.preprocessing       import image
from   keras.preprocessing.image import ImageDataGenerator
from   keras.engine.topology     import get_source_inputs
from   keras.utils.data_utils    import get_file
from   keras.applications.imagenet_utils import decode_predictions, _obtain_input_shape

root_prefix = "C:/Users/mislam/Final Project/demo"

data_Train=pd.read_csv("C:/Users/mislam/Final Project/data/ISIC-2017_Training_Part3_GroundTruth.csv")
data_Valid=pd.read_csv("C:/Users/mislam/Final Project/data/ISIC-2017_Validation_Part3_GroundTruth.csv")
data_Test=pd.read_csv("C:/Users/mislam/Final Project/data/ISIC-2017_Test_v2_Part3_GroundTruth.csv")

data_Train=data_Train.iloc[0:2000,1]
data_Valid=data_Valid.iloc[0:150,1]
data_Test=data_Test.iloc[0:600,1]
print (data_Train)
print (data_Valid)
print (data_Test)

image_width  = 299
image_height = 299
image_size = (image_width, image_height)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,cval=0.05,
    horizontal_flip=True,vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
        '%s/mytrain' % (root_prefix),
        target_size=image_size,  # all images will be resized to 299x299
        batch_size=16,
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(
        '%s/myvalid' % (root_prefix),
        target_size=image_size, 
        batch_size=16,
        class_mode='binary')
		
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
        '%s/mytest' % (root_prefix),
        target_size=image_size, 
        batch_size=1,
        class_mode='binary')

prefix = "https://github.com/fchollet/deep-learning-models/releases/download/v0.4/"
TF_WEIGHTS_PATH        = '%s/xception_weights_tf_dim_ordering_tf_kernels.h5'       % (prefix)
TF_WEIGHTS_PATH_NO_TOP = '%s/xception_weights_tf_dim_ordering_tf_kernels_notop.h5' % (prefix)


# Instantiate the Xception architecture, optionally loading weights pre-trained on ImageNet. 
def Xception(include_top=True, weights='imagenet', input_tensor=None):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    x = Conv2D(32, 3, 3, subsample=(2, 2), bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, 3, 3, bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block2_pool')(x)
    x = merge([x, residual], mode='sum')

    residual = Conv2D(256, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block3_pool')(x)
    x = merge([x, residual], mode='sum')

    residual = Conv2D(728, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block4_pool')(x)
    x = merge([x, residual], mode='sum')

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)
        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = merge([x, residual], mode='sum')

    residual = Conv2D(1024, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block13_pool')(x)
    x = merge([x, residual], mode='sum')

    x = SeparableConv2D(1536, 3, 3, border_mode='same', bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, 3, 3, border_mode='same', bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)

    return model


base_model = Xception(include_top=False, weights='imagenet')
x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid', name='output')(x)

top_num = 20
model   = Model(input=base_model.input, output=x)

for layer in model.layers[:-top_num]:
    layer.trainable = False

for layer in model.layers[-top_num:]:
    layer.trainable = True

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
best_model = ModelCheckpoint("xception_best.h5", monitor='val_loss', verbose=0, save_best_only=True)
print(model.summary())
from keras.callbacks import ModelCheckpoint, TensorBoard
model.fit_generator(
        train_generator,
        samples_per_epoch=2000,nb_epoch=100,validation_data=validation_generator,
        nb_val_samples=150,
        callbacks=[best_model])

result=model.evaluate_generator(test_generator)
print("Accuracy on Validation Image: ", result[1])
# predict = model.predict_generator(test_generator)
# auc=roc_auc_score(data_Test, predict)
# print ("AUC Score on Test Image: ",auc)
del model

from keras.models import load_model

model = load_model('xception_best.h5')
result=model.evaluate_generator(test_generator)
# auc=roc_auc_score(data_Test, predict)
print("Accuracy on Validation Image: ", result[1])




