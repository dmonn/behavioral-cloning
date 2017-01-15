# Project 3 - Behavioral Cloning

This is a short walkthrough of the training approach used to solve the problem of Udacity's third project "Behavioral Cloning".

The logic is based in *model.py* while *drive.py* handles the driving.

## Data Collection

A simulator is provided. I used a XBOX360 controller and a custom-made 50hz simulator (included in the slack channel) to collect ~40'000 examples. Some training showed that the data I have collected is not accurate enough. The behaviour of the car was "wobbly" in general and the car couldn't drive a straight line.

I tried collecting more data with smoother steering and a lot of straight driving examples. I've also added "rescuing" examples, which means I drove close to the edge and then directed the car towards the middle of the street. After another test, I've found that the behavioural of the car was still wobbly and unsafe.

After the first test with the provided Udacity dataset, the car behaved very differently and even passed the bridge on the first track. Even though the dataset is about 4-6x smaller than my previously used datasets, this gave me a much better performance. The training also went a lot quicker.

## Preprocessing

I'm cropping the given images to the following dimensions.
![Cropping](https://raw.githubusercontent.com/dmonn/SDCND/master/CarND-Behavioral-Cloning/figure_1.png?token=AGHDdxZyWnQlpTwrPvGW1kZzkVPq-VXDks5Yfy7nwA%3D%3D "Cropping of image data")

I'm also resizing the image to (66, 200) to match nvidia's paper and normalize it. Because I'm using a fairly small dataset, I'm using data augmentation, described very well by Vivek Yadav, to generate additional data. I'm using left, right and center images randomly with a steering shift value and I'm also flipping the images in 50% of the cases.

To cancel out "sluggish" behaviour, I'm also removing about 70% of all small/zero steering angles and I'm shuffling/random picking my batches.

I'm splitting my dataset into train/validation set with a factor of 0.9 (which means 90% is training, 10% is validation/test)

## Network / Training

The data which is loaded directly from my drive is feed into a Keras `fit_generator`. My generator picks a random batch of my dataset, picks one of the images (left, center, right) and might flip the image. Then the generator yields the batch and feeds it to the network. This continues until the whole dataset is used.

I'm using a model based on [Nvidia's paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). To reduce overfitting, I have added several dropout layers to the network.

![Network](https://raw.githubusercontent.com/dmonn/behavioral-cloning/master/model-visualization.png?token=AGHDd-jb9QsCb1Rw-pTEwTTm7wknuCmyks5YhQ0LwA%3D%3D "Network Visualization")

The network consists of five convolutional layers, followed by three fully connected layers. I have added Dropout Layers and SpatialDropout Layers to prevent overfitting. The `model.summary()` command prints the following output:

```
____________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
lambda_2 (Lambda)                (None, 66, 200, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 100, 24)   1824        lambda_2[0][0]
____________________________________________________________________________________________________
spatialdropout2d_1 (SpatialDropo (None, 33, 100, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 50, 36)    21636       spatialdropout2d_1[0][0]
____________________________________________________________________________________________________
spatialdropout2d_2 (SpatialDropo (None, 17, 50, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 23, 48)     43248       spatialdropout2d_2[0][0]
____________________________________________________________________________________________________
spatialdropout2d_3 (SpatialDropo (None, 7, 23, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 21, 64)     27712       spatialdropout2d_3[0][0]
____________________________________________________________________________________________________
spatialdropout2d_4 (SpatialDropo (None, 5, 21, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 19, 64)     36928       spatialdropout2d_4[0][0]
____________________________________________________________________________________________________
spatialdropout2d_5 (SpatialDropo (None, 3, 19, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 3648)          0           spatialdropout2d_5[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 3648)          0           flatten_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           364900      dropout_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_2[0][0]
====================================================================================================
Total params: 501,819
Trainable params: 501,819
Non-trainable params: 0
____________________________________________________________________________________________________

```

I'm using lambda layers to normalize the data and resize it. This has the effect that I don't need to add these things in *drive.py* because they are already included in the network itself.

I'm using an Adam optimizer with a learning rate 0.001. This very small learning rate allows me to get a generalized result over 50 epochs.

## Testing / Autonomous Driving

After my model always had a different behaviour after re-training, even though I didn't change anything, I realized that the model always sets its weights a little bit different. This was enough for the car to sometimes go straight off the track or crash somewhere later. Also the behaviour on my main machine (GTX980M, i7, 16GB RAM) and my Macbook (Intel Iris, i7, 8GB RAM) was fairly different. I've decided to run the autonomous mode on lowest settings with a resolution of 1024x768px on my main machine.

To test out all the weights, I trained the model a lot of time (about 8 times per change) and then choose the best performing model.

## Second Submission

After the first submission, I added a translation function, provided by Vivek Yadav and increased  the learning rate. I removed the vertical cutting, so that I only cut from top and bottom, to add more detail to the camera images.

 The car now drives fully autonomously without leaving the lane.
