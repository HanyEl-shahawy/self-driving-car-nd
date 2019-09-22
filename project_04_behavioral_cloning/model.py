import csv, argparse, cv2
import json
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Conv2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import ModelCheckpoint
#from preprocess import *
from preprocess_small import *

def small_model():
    # nvidia model
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=INPUT_SHAPE))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(16, (5, 5), strides=(2, 2), activation='elu'))
    # drop out layer
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(16, (3, 3), strides=(2, 2), activation='elu'))
    # drop out layer
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1024, activation='elu'))
    # drop out layer
    model.add(Dropout(0.3))    
    model.add(Dense(512, activation='elu'))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


def nvidia_model(keep_prob):
    # nvidia model
    model = Sequential()
    # normalization
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    # drop out layer
    model.add(Dropout(keep_prob))
    model.add(Flatten())
#     model.add(Dense(1164, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
#     learning_rate = 0.00001
#     optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
#     model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.compile(optimizer="adam", loss="mse")
    return model

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    parser = argparse.ArgumentParser(description='Project Behavioral Cloning')
    parser.add_argument('-c', help='csv file',dest='csv',type=str,default='driving_log.csv')
    parser.add_argument('-b', help='batch size', dest='batch_size',type=int, default=32)
    parser.add_argument('-e', help='epoch number', dest='epoch',type=int, default=5)
    parser.add_argument('-v', help='validation size %',dest='valid_size',type=float, default=0.2)
    parser.add_argument('-k', help='drop out %',dest='keep_prob',type=float, default=0.2)
    parser.add_argument('-n', help='out network name',dest='name', default='model.h5')
    parser.add_argument('-r', help='retrain',dest='retrain', type=int, default=0)
    args = parser.parse_args()
    
    # read the data
    data = pd.read_csv(args.csv)
    # shuffle the data
    data_shuffled = data.sample(frac=1).reset_index(drop=True)
    data_shuffled.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'] 

    # split training and validation samples
    training_size = 1 - args.valid_size
    rows_training = int(data_shuffled.shape[0]*training_size)
    training_data = data_shuffled.loc[0:rows_training-1]
    validation_data = data_shuffled.loc[rows_training:]
    # create model and generators
    print('args.retrain: ',args.retrain)
    if args.retrain == 0:
        print('new')
#         model = nvidia_model(args.keep_prob)
        model = small_model()
    else:
        print('loaded')
        model = load_model(args.name)
    training_generator = data_generator(training_data, args.batch_size)
    valid_generator = data_generator(validation_data, args.batch_size)
#     steps_per_epoch = (training_data.shape[0] // args.batch_size) * 10
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 mode='auto')
    model.fit_generator(training_generator, steps_per_epoch=4000, nb_epoch=args.epoch, validation_data=valid_generator, validation_steps=300, callbacks=[checkpoint], verbose=1)

if args.retrain == 0:
# Save the model as json after each epoch.
    model.save(args.name)    
    print('model saved')
else:
    # Update the model as json after each epoch.
    model.save('new_' + args.name)    
    print('model updated')

