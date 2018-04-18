# **Behavioral Cloning** 

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./static/nvidia-cnn.png "Model Visualization"
[image6]: ./static/regular.png "Normal Image"
[image7]: ./static/flipped.png "Flipped Image"
[image7]: ./static/correction.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py models/final.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training, data preparation and saving the convolution neural network. 

The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is derived from the NVIDIA self driving car model from the class.  It contains:
 * Normalization using a Keras lambda layer
 * Cropping
 * 5 convolutional layers (with RELU activation)
 * 5 dense layers (with RELU activation) including the output layer for the steering angle  

#### 2. Attempts to reduce overfitting in the model

1. I did split into a training and validation set to ensure that the model was not overfitting.
2. My model didn't appear to overfit too bad, as such, I didn't implement any drop out or regularization.  
3. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

My training data consisted of two primary components:
1. Lap driving - This is data where I drove the simulator around the track to ensure good driving behavior and what should be expected of the self
driving car.  I took 3 laps around normally (i.e. counter-clockwise) and one lap backwards.
2. Correction driving - Based on recommendations in the course, I did corrections in tough areas (i.e. dirt roads, sharp curves, bridge, etc.)
where I'd go off to the side and record my re-entry. 

After this, my car performed well! Here's some sample images.

Regular
![regular](https://github.com/jhevrin2/CarND-Behavioral-Cloning-P3/blob/master/static/regular.jpg "regular")

Flipped
![flipped](https://github.com/jhevrin2/CarND-Behavioral-Cloning-P3/blob/master/static/flipped.jpg "flipped")

Correction
![correction](https://github.com/jhevrin2/CarND-Behavioral-Cloning-P3/blob/master/static/correction.jpg "correction")

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I tried an iterative approach to getting the final model working:
1. I attempted to just use the center images.  This did poorly and only made it around one turn.
2. I incorporated in the left and right images, a small adjustment factor and cropping.  This made it to the bridge.
3. I noticed my car jerked as it drove, so I used the mouse and recollected data of smoother driving.  This made it over the bridge.
4. I flipped the images to double my data.  This made it to the dirt road.
5. I added a stronger adjustment factor, more corrections and increased the epoches.  This finished the course successfully!

Throughout the tests, I leveraged the NVIDIA driving convolutional neural network.

#### 2. Final Model Architecture

As touched on above, here is the network architecture in picture form:

![nvidia_network](https://github.com/jhevrin2/CarND-Behavioral-Cloning-P3/blob/master/static/nvidia-cnn.png "nvidia-cnn")

#### 3. Creation of the Training Set & Training Process

See above on training data and my solution design.  Lots of attempts, but slowly made it further each time.