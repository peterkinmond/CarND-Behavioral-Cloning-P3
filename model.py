import csv
import cv2
import numpy as np


# Import driving image data
lines = []
with open('./data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
# for line in lines[0:2]:
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3]) # steering angle
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

images = images + augmented_images
measurements = measurements + augmented_measurements

print("There are {0} images and {1} measurements".format(len(images), len(measurements)))

# Create training set (validation set created by Keras function below)
X_train, y_train = images, measurements
X_train = np.array(images)
y_train = np.array(measurements)

# Create the model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Crop the irrelevant part of the images (top tree line, bottom car pixels)
model.add(Cropping2D(cropping=((80, 20), (0, 0)), input_shape=(160,320,3)))
# pixel_normalized = pixel / 255.0 # shifts values into 0 to 1 range
# pixel_mean_centered = pixel_normalized - 0.5 # shifts values into -0.5 to 0.5 range
model.add(Lambda(lambda x: x / 255.0 - 0.5))

model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# Train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, epochs=4, shuffle=True)

# Save the model
model.save('model.h5')
exit()

