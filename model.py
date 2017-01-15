import pickle
import csv
import matplotlib.pyplot as plt
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, Input, Lambda, SpatialDropout2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
import cv2
import numpy as np
import pandas as pd
import h5py
import sys

K.set_image_dim_ordering("tf")

BATCH_SIZE = 64
EPOCHS = 10

DEBUGING_FLAG = False

DATA_PATH = "/home/dmonn/Downloads/data/data"
LABEL_PATH = "{}/driving_log.csv".format(DATA_PATH)

def import_csv():
    """
    Saving the CSV data in a array
    """
    data = []
    with open(LABEL_PATH) as FILE:
        reader = csv.reader(FILE)
        for i in reader:
            data.append(i)

    return data

def process_img(img):
    """
    Load image and crop
    """
    img = "{}/{}".format(DATA_PATH, img)
    img = plt.imread(img)[60:135, : ]

    if DEBUGING_FLAG:
        # Show image if Debug Flag is enabled
        plt.imshow(img)
        plt.show()
        sys.exit("Ending preprocessing here; not done.")

    return img

def get_batch(data):
    """
    Randomly select batch
    """
    indices = np.random.choice(len(data), BATCH_SIZE)
    return data.sample(n=BATCH_SIZE)

def randomize_image(data, value):
    """
    Randomize between left, center and right image
    And add a shift
    """
    random = np.random.randint(4)
    if (random == 0):
        path_file = data['left'][value].strip()
        shift_ang = .25
    if (random == 1 or random == 3):
        # Twice as much center images
        path_file = data['center'][value].strip()
        shift_ang = 0.
    if (random == 2):
        path_file = data['right'][value].strip()
        shift_ang = -.25

    return path_file,shift_ang

def generate_train(data):
    """
    Train data generator
    """
    obs = 0
    while 1:
        batch = get_batch(data)
        features = np.empty([BATCH_SIZE, 75, 320, 3])
        labels = np.empty([BATCH_SIZE, 1])

        for i, value in enumerate(batch.index.values):
            x, shift = randomize_image(data, value)
            x = process_img(x)

            x = x.reshape(x.shape[0], x.shape[1], 3)
            features[i] = x

            # Add shift to steer
            y = float(data['steer'][value]) + shift

            random = np.random.randint(1)

            # Flip image in 50% of the cases
            # Thanks to Vivek Yadav for the idea
            if (random == 0):
                x = np.fliplr(x)
                y = -y

            labels[i] = y

        x = np.array(features)
        y = np.array(labels)
        obs += len(x)
        yield x, y

def generate_valid(data):
    """
    Validation Generator
    """
    while 1:
        for i_line in range(len(data)):
            data = data.iloc[[i_line]].reset_index()
            x = process_img(data['center'][0])
            x = x.reshape(1, x.shape[0], x.shape[1], 3)
            y = data['steer'][0]
            y = np.array([[y]])
            yield x, y

def remove_low_steering(data):
    """
    Remove about 70% of steering values below 0.05
    """
    ind = data[abs(data['steer'])<.05].index.tolist()
    rows = []
    for i in ind:
        random = np.random.randint(10)
        if random < 7:
            rows.append(i)

    data = data.drop(data.index[rows])
    print("Dropped {} rows with low steering".format(len(rows)))
    return data

def nvidia(img):
    """
    Model based on Nvidia paper
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """

    shape = (img[0], img[1], 3)

    model = Sequential()

    def process(img):
        import tensorflow as tf
        # img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize_images(img, (66, 200))
        return img

    model.add(Lambda(process, input_shape=shape))

    model.add(Lambda(lambda x: x/255.-0.5))
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model

# 0 = center
# 1 = left
# 2 = right
# 3 = steering angle

for i in range(5):
    # Train the whole thing 5 times
    # Load data
    data = pd.read_csv(LABEL_PATH, index_col=False)
    data.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']

    img = process_img(data['center'][900].strip())

    model = nvidia(img.shape)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')

    # Shuffle data
    data_shuffle = data.reindex(np.random.permutation(data.index))

    # Split data on a multiple of BATCH SIZE
    split = (int(len(data_shuffle) * 0.9) // BATCH_SIZE) * BATCH_SIZE
    train_data = data[:split]

    train_data = remove_low_steering(train_data)

    val_data = data[split:]
    new_val = (len(val_data) // BATCH_SIZE) * BATCH_SIZE
    val_data = val_data[:new_val]

    samples_per_epoch = len(train_data) - BATCH_SIZE

    values = model.fit_generator(generate_train(train_data), samples_per_epoch=samples_per_epoch, nb_epoch=EPOCHS, validation_data=generate_train(val_data), nb_val_samples=len(val_data))

    model_rep = model.to_json()

    # Save data
    with open('model-' + str(i) + '.json', 'w') as f:
        json.dump(model_rep, f)

        model.save_weights('./model-'+ str(i) +'.h5')

        print("It's saved now")
