# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:10:39 2018

@author: Camila
"""



from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
#from collections import Counter
from keras.optimizers import Adam
import matplotlib.pyplot as plt

###even it out
#counter = Counter(training_set.classes)                          
#max_val = float(max(counter.values()))       
#class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}  
#counter = Counter(test_set.classes)                          
#max_val = float(max(counter.values()))       
#class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}  
#classifier.fit_generator(training_set,
#                         steps_per_epoch = 1070,
#                         epochs = 30,
#                         validation_data = test_set,
#                         validation_steps = 2000)

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'C:\\Users\\Camila\\Documents\\EDP\\Dataset_test\\train'
validation_data_dir = 'C:\\Users\\Camila\\Documents\\EDP\\Dataset_test\\validation'
nb_train_samples = 500
nb_validation_samples = 800
epochs = 50
batch_size = 16
#class_weights={0:1,1:2}

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    zoom_range=0.2,
    width_shift_range = 0.2,
    horizontal_flip=True,
    samplewise_center=True,
    samplewise_std_normalization=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#counter = Counter(train_generator.classes)
#max_val = float(max(counter.values()))
#class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
#    class_weight=class_weights,
    shuffle=True)

# list all data in history
#print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#save the model
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")