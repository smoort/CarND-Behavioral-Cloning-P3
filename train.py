import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os.path

samples = []

###  Load data
with open('.\data\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
#        if line[3] != '0':
        samples.append(line)
        
print('Number of records in input file = ', len(samples))
print('File read complete')



"""
###  Visualize Data

steering_angles = []
for sample in samples:
    steering_angle = float(sample[3])
    steering_angles.append(steering_angle)
print(len(steering_angles))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('VISUALIZING TRAINING CLASS DISTRIBUTION')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
n, bins, patches = plt.hist(steering_angles, bins=21)
plt.title("Streeing Anlge Histogram")
plt.xlabel("Streering Angle")
plt.ylabel("Frequency")
plt.axis([-1, 1, 0, 10000])
plt.grid(True)
plt.show()



###  Augument Data

augumented_images, augumented_measurements = [], []
for sample in samples:
    augumented_images.append(image)
    augumented_measurements.append(measurement)
    augumented_images.append(cv2.flip(image,1))
    augumented_measurements.append(measurement * -1.0)

AX_train = np.array(augumented_images)
Ay_train = np.array(augumented_measurements)

print('Number of records after augumentation = ', len(samples))
print('Data augumentation complete')

###  Preprocess Data

"""

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
#        X_train = np.zeros(shape=(32,160,320,3))
#        y_train = np.zeros(shape=(32,1))
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]
                current_path = '.\data\IMG\\' + filename
                image = cv2.imread(current_path)
                images.append(image)
                measurement = float(batch_sample[3])
                measurements.append(measurement)
                
                images.append(cv2.flip(image,1))
                measurements.append(measurement * -1.0)

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

"""
# Create a model
model = Sequential()
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
#model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
#model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

"""

# Nvidia model
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
    model.load_weights('my_model_weights.h5')

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
