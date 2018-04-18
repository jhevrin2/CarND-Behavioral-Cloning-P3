import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Lambda, Convolution2D, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D


def read_data(filename):
    """
    Imports data file generated from the simulator into a list
    :param filename: File name to be read in
    :return: array of samples data
    """
    samples = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


def generator(samples, batch_size=32):
    """
    Creates a generator to feed batches into the Keras model
    :param samples: Samples list from the simulator
    :param batch_size: Size of the batches
    :return: A generator
    """
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                images, angles = process(images, angles, batch_sample)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def process(images, angles, batch_sample):
    """
    Processes all of the images into the images and angles array
    :param images: List of images
    :param angles: List of angles
    :param batch_sample: Batch of samples
    :return: Images and angles arrays
    """
    center, left, right = read_images(batch_sample)
    angle = float(batch_sample[3])
    center_angle, left_angle, right_angle = create_angles(angle)
    images, angles = append_values(images, angles, center, center_angle, left, left_angle, right, right_angle)
    return images, angles


def append_values(images, angles, center, center_angle, left, left_angle, right, right_angle):
    """
    Appends all of the angles and images for a batch
    :return: Images and angles arrays
    """
    images.append(center)
    images.append(left)
    images.append(right)
    angles.append(center_angle)
    angles.append(left_angle)
    angles.append(right_angle)
    # Append flipped iamges
    flipped_center = np.fliplr(center)
    flipped_left = np.fliplr(left)
    flipped_right = np.fliplr(right)
    images.append(flipped_center)
    images.append(flipped_left)
    images.append(flipped_right)
    angles.append(-center_angle)
    angles.append(-left_angle)
    angles.append(-right_angle)
    return images, angles


def create_angles(angle, correction=0.25):
    """
    Creates the angles for the three image types
    :param angle: Center angle
    :param correction: Correction factor
    :return: Angles for center, left and right
    """
    center = angle
    left = angle + correction
    right = angle - correction
    return center, left, right


def read_images(batch_sample):
    """
    Reads in the images for center, left, right
    :param batch_sample: Sample
    :return: Images for center, left, right
    """
    center = cv2.imread(batch_sample[0])
    left = cv2.imread(batch_sample[1])
    right = cv2.imread(batch_sample[2])
    return center, left, right


def build_model():
    """
    Model declaration based off of the NVIDIA self driving car
    :return: Keras model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse', optimizer='adam')
    return model


if __name__ == "__main__":
    laps = read_data("/Users/mwk1/Desktop/laps/driving_log.csv")
    corrections = read_data("/Users/mwk1/Desktop/corrections/driving_log.csv")
    samples = laps + corrections

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = build_model()
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                        nb_val_samples=len(validation_samples), nb_epoch=10)
    model.save('models/final.h5')
