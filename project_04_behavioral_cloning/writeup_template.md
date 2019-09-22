# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./arch.png "Architecture"
[image2]: ./original.png "Original"
[image3]: ./crop.png "Cropped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create model and main entry to train the model
* drive.py for driving the car in autonomous mode
* preprocess_small.py contains all preprocessing functions done to the image and augmentation techniques used. It also has the data generator function
* simple_model_sep4000.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py simple_model_sep4000.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network along with preprocess_small.py. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5, 3x3 and 2x2 filters sizes and depths between 16 and 32 (model.py lines 14-34) 

The model includes ELU layers to introduce nonlinearity (code line 18), and the data is normalized in the model using a Keras lambda layer (code line 14). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used Udacity data set but added augmentation and used right and left cameras as shown in following sections. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model is similar to [Augmentation_based model](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.d779iwp28) and will be discussed in details in the following sections

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

To combat the overfitting, i added drop out layers. I then preprocessed the input image to remove the the hood of the car and the sky and resized it.

Then I augmented the data to have the network trained on various conditions thus better results. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track or hit the wall. To improve the driving behavior in these cases, I increased epochs and number of steps per epoch and also collected more data in such specific conditions.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture
I tried first Nvidia model with full architecture. That was lots of parameters which needed training so i decided to go for a simpler model which needs less training.

Eventually, the model is as follows:
1- Normalization layer
2- Convolution: 5x5, filter: 32, strides: 2x2, activation: ELU
3- Convolution: 5x5, filter: 16, strides: 2x2, activation: ELU
4- Drop out layer with probability(0.4)
5- Max pooling: 2x2
6- Convolution: 3x3, filter: 16, strides: 2x2, activation: ELU
7- Drop out layer with probability(0.4)
8- Fully connected: neurons: 1024, activation: ELU
9- Drop out layer with probability(0.3)
10- Fully connected: neurons: 512, activation: ELU
11- Fully connected: neurons: 1, activation: ELU
(model.py lines 12-34) 

Here is a visualization of the architecture 
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I used data set provided from Udacity. Here is a sample image ![alt text][image2].
I cropped the hood of the car and the sky since they are not contributing to the network output, as shown in this image ![alt text][image3]

I used left and right images to train the network more, i added offset to steering angle to simulate the car driving behavior from this camera's point of view.

To augment the data sat, I flipped images and angles since the track has prevailing left turns so that the network gets evenly balanced data. I also reduced image brightness randomly by converting it to HSV color map then multiply V component by a ratio.

I trained the network on the fly; meaning that every step there will be a random choice of which camera image to choose, is it flipped or not and how much brightness is reduced. 

I randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 from try and error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
