# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:56:33 2018

@author: Camila
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model2.h5'
train_data_dir = 'C:\\Users\\Camila\\Documents\\EDP\\Dataset_test\\train'
validation_data_dir = 'C:\\Users\\Camila\\Documents\\EDP\\Dataset_test\\validation'
nb_train_samples = 336+336
nb_validation_samples = 800
epochs = 50
batch_size = 16
# np.save("example") np.load("example.npy")

def save_bottlebeck_features():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
        rotation_range=40,
        zoom_range=0.2,
        width_shift_range = 0.2,
        horizontal_flip=True,
        samplewise_center=True,
        samplewise_std_normalization=True)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save('bottleneck_features_train.npy',
            bottleneck_features_train)
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array(
        [0] * 336 + [1] * 336)

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array(
        [0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()