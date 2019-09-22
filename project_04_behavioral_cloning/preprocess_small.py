import cv2, os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from io import BytesIO
import base64


import pandas as pd


# define random seed 
RANDOM_SEED = 0
# input shape to network
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 64, 64, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def crop(image):
    # remove sky & car front
    return image[55:135, :, :]


def resize(image):
    # resize image to INPUT_SHAPE
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

def preprocess(image):
    # run pre-processing steps
    image = crop(image)
    image = resize(image)
    image = image.astype(np.float32)
    return image

def flip(image, steering_angle):
    # randomly flip image and steering
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def reduce_brightness(image):
    # convert image to hsv then decrease brightness by factor
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # to avoid complete darkness, 0.25 is added
    ratio = 0.10 + np.random.uniform()
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def translate_image(image,steer,trans_range=100):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image.shape[1],image.shape[0]))    
    return image_tr,steer_ang

def augment_and_preprocess(csv_row):
    # randomly choose image from input csv row
    steering = csv_row['steering']
    # randomly choose the camera to take the image from
    camera = np.random.choice(['center', 'left', 'right'])
    # add offset to steering angle to adjust for left anf right cameras
    if camera == 'left':
        # read image
        img = load_img(csv_row['left'].strip())
        steering += 0.25
    elif camera == 'right':
        # read image
        img = load_img(csv_row['right'].strip())
        steering -= 0.25
    else:
        # read image
        img = load_img(csv_row['center'].strip())

    img = img_to_array(img)
    # flip
    img, steering = flip(img, steering)
    # reduce brightness
    img = reduce_brightness(img)
    # preprocess
    img = preprocess(img)
    return img, steering

def data_generator(data, batch_size):
    # generate trainging image from csv file
    num_data = data.shape[0]
    batches_per_epoch = num_data // batch_size
    i = 0  
    while True:
        start = i*batch_size
        end = start+batch_size - 1
        X = np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.float32)
        Y = np.zeros((batch_size,), dtype=np.float32)
        j = 0
        for index, row in data.loc[start:end].iterrows():
            X[j], Y[j] = augment_and_preprocess(row)
            j += 1
        i += 1
        if i == batches_per_epoch - 1:
            i = 0
        yield X, Y
            
# if __name__ == '__main__':
#     img = load_img('./IMG/center_2016_12_01_13_41_15_685.jpg')
#     img = img_to_array(img)
#     print(img.shape)
#     save_img('original.png',img)
#     img, _ = translate_image(img, 0.5)
#     save_img('trans.png',img)
#     img = flip(img, 0.5)
#     save_img('flip.png',img)
#     img_crop = crop(img)
#     print(img_crop.shape)
#     save_img('crop.png',img_crop)
#     img_crop_resize = resize(img_crop)
#     print(img_crop_resize.shape)
#     save_img('crop_resized.png',img_crop_resize)
#     img_pre = rgb2yuv(img_crop_resize)
#     save_img('crop_resized_yuv.png',img_pre)
#     img_reduced_brightness = reduce_brightness(img_pre)
#     save_img('reduced_brightness.png',img_reduced_brightness)