import cv2
import csv
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

import keras
print(keras.__version__)

from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Sequential
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import Adam
from keras.regularizers import l2


DATA_ROOT = 'data/'
DATA_DIRS = [
    'data',
    'track1_center',
    'track1_center2',
    'track1_center_ccw',
    'track1_curve',
    'track1_curve_ccw',
    'track1_bridge',
    'track1_patch',
    'track1_recover',
    'track1_recover_ccw',
    # TODO: include track2 data cause regression in track1. Exclude them from training for now.
    #'track2_center',
    #'track2_center_ccw'
]

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 320


def read_image(x):
    """Read image from directory and convert to RGB image"""
    # Convert the image path record in csv to the real path.
    h1, t1 = os.path.split(x)
    h2, t2 = os.path.split(h1)
    _, t3 = os.path.split(h2)
    image_path = os.path.join(DATA_ROOT, t3, t2, t1)
    # cv2 imread is BRG, need to convert to RGB.
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def random_flip(image, angle):
    """Flip the image with 0.5 probability""" 
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    return image,angle 


def random_shadow(image):
    """Add random shadow to the image, inlightened by https://github.com/naokishibuya"""
    #(x1, y1) and (x2, y2) forms a line cut image to top and bottom.
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    # xm, ym gives all the locations of the image
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    # Assign one side with 1 and the other side with 0 in the mask.
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
    # Randomly choose one side and darken it by adjust its Saturation
    cond = mask == np.random.randint(2)
    s_multiplier = np.random.uniform(low=0.2, high=0.6)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_multiplier
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
    

def random_brightness(image, multiplier=0.4):
    """Adjust the brightness randomly in the range [0.8, 1.2]"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + multiplier * (np.random.rand() - 0.5)    
    hsv[:,:,2] = hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(image, angle):
    """Augment the image. It goes through multiple steps, and each step add random distortion to the image"""
    image, angle = random_flip(image, angle)
    if np.random.rand() < 0.5:
        image = random_shadow(image)
    if np.random.rand() < 0.5:
        image = random_brightness(image)
    return image, angle


def generator(samples, is_training, batch_size=64):
    """Generator which provide batch to model"""
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center = batch_sample[0]
                angle = float(batch_sample[3])
                image = read_image(center)
                # In training mode, augment the image with random distortion.
                if is_training:
                    image, angle = augment(image, angle)
                
                images.append(image)
                angles.append(angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

def get_model():
    """
    A modified version of nVidia network.
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

    NOTE: dropout is replaced with batch normalization, since the later has better performance in experiment.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x/127.5)-1.0, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(64,3, 3, subsample=(1,1), border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(64,3, 3, subsample=(1,1), border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1))
    return model


def main():
    """Main function to load data and train modea"""
    # Read samples from data directories.
    samples = []
    for data_dir in DATA_DIRS:
        with open(os.path.join(DATA_ROOT, data_dir, 'driving_log.csv')) as csvfile:
            reader = csv.reader(csvfile)
            header = True
            for line in reader:
                if header:
                    header = False
                else:
                    # Special handling for data/ folder
                    if data_dir == 'data':
                        for i in range(0, 3):
                            line[i] = os.path.join('data', line[i])
                    samples.append(line)

    # Split training data and validation data with 80/20.
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, is_training=True)
    validation_generator = generator(validation_samples, is_training=False)

    # Train model.
    model = get_model()
    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
    #model.compile(loss='mse', optimizer=Adam(lr=1e-4))
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                        validation_data=validation_generator, nb_val_samples=len(validation_samples),
                        nb_epoch=10, callbacks=[checkpoint])


if __name__ == '__main__':
    main()
