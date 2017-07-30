import csv
import cv2
import numpy as np


# Import driving image data
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
# for line in lines[0:2]:
for line in lines:
    filename = line[0]
    image = cv2.imread(filename)
    images.append(image)
    measurement = float(line[3]) # steering angle
    measurements.append(measurement)

print("There are {0} images and {1} measurements".format(len(images), len(measurements)))

# Create training set (validation set created by Keras function below)
X_train, y_train = images, measurements
X_train = np.array(images)
y_train = np.array(measurements)

# Create the model
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

# Train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

# Save the model
model.save('model.h5')

