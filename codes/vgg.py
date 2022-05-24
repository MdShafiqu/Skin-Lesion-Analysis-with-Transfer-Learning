import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Flatten, Dense,Dropout
from keras.callbacks import Callback, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


model_vgg16_conv = VGG16(weights='imagenet', include_top=False)


for layer in model_vgg16_conv.layers:
    layer.trainable = False

IMG_WIDTH, IMG_HEIGHT = 224, 224
TRAIN_DATA_DIR = 'C:/Users/mislam/Final Project/demo/mytrain'
VALIDATION_DATA_DIR = 'C:/Users/mislam/Final Project/demo/myvalid'
MODEL_WEIGHTS_FILE = 'C:/Users/mislam/CatDog_Kaggle/vgg16_weights.h5'
NB_TRAIN_SAMPLES = 2000
NB_VALIDATION_SAMPLES = 150
NB_EPOCH = 1

input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
output_vgg16_conv = model_vgg16_conv(input)
x = Flatten()(output_vgg16_conv)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax')(x)
model = Model(input=input, output=x)

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=32,
        class_mode='categorical')

best_model = ModelCheckpoint("vgg_best.h5", monitor='val_loss', verbose=0, save_best_only=True)
history = model.fit_generator(
        train_generator,
        samples_per_epoch=NB_TRAIN_SAMPLES/8,
		validation_data=validation_generator,
        nb_val_samples=150/8,
        nb_epoch=NB_EPOCH,callbacks=[best_model])

result=model.evaluate_generator(validation_generator)
print("Accuracy on Validation Image: ", result[1])
model.save('vgg_16.h5')


