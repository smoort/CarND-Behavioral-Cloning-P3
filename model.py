import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os.path

samples = []

###  Load data from multiple folders that hold training images

path = './data/'
#with open('.\data1\driving_log.csv') as csvfile:
with open(path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #if line[3] != '0':
        img_path = line[0]
        filename = img_path.split('\\')[-1]
        #new_path = path + 'IMG\\' + filename
        new_path = path + 'IMG/' + filename
        line[0] = new_path
        if line[3] != '0':
            samples.append(line)

path = './data1/'
#with open('.\data1\driving_log.csv') as csvfile:
with open(path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #if line[3] != '0':
        img_path = line[0]
        filename = img_path.split('\\')[-1]
        #new_path = path + 'IMG\\' + filename
        new_path = path + 'IMG/' + filename
        line[0] = new_path
        samples.append(line)
sklearn.utils.shuffle(samples)
        
path = './data2/'
#with open('.\data1\driving_log.csv') as csvfile:
with open(path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #if line[3] != '0':
        img_path = line[0]
        filename = img_path.split('\\')[-1]
        #new_path = path + 'IMG\\' + filename
        new_path = path + 'IMG/' + filename
        line[0] = new_path
        samples.append(line)
sklearn.utils.shuffle(samples)

print('Number of records in input file = ', len(samples))
print('File read complete')

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('Training data size after split = ', len(train_samples))
print('Validation data size after split = ', len(validation_samples))


images = []
measurements = []

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:

                # create adjusted steering measurements for the side camera images
                correction = 0.2
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                
                image_center = cv2.imread(batch_sample[0])
                image_left = cv2.imread(batch_sample[1])
                image_right = cv2.imread(batch_sample[2])
                
 
               # add images and angles to data set 
                images.append(image_center)
                #images.append(image_left)
                #images.append(image_right)

                measurements.append(steering_center)
                #measurements.append(steering_left)
                #measurements.append(steering_right)
                
                # Augument data by flipping the image and adjusting steering angle accordingly
                images.append(cv2.flip(image_center,1))
                measurements.append(steering_center * -1.0)
                #images.append(cv2.flip(image_left,1))
                #measurements.append(steering_left * -1.0)
                #images.append(cv2.flip(image_right,1))
                #measurements.append(steering_right * -1.0)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

###  Visualize sample data to confirm that all is okay

#plt.imshow(X_train[100])
#plt.show()

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D


# Create a model - Adopted the Nvidia model

model = Sequential()
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

### Load weights if available
if os.path.exists('my_model_weights.h5'):
    print('loading weights from saved file')
    model.load_weights('my_model_weights.h5')

### Model Execution
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history_object = model.fit_generator(train_generator, steps_per_epoch= (len(train_samples)/32), validation_data=validation_generator, 
            validation_steps=(len(validation_samples)/32), epochs=3, verbose=1)

### Save the model
model.save('model.h5')

### Save the model weights
model.save_weights('my_model_weights.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()