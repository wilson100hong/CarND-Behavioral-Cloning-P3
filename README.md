[//]: # (Image References)

[image1]: ./images/nvidia_model.png "nVidia model"
[image2]: ./images/data_distribution.png "steering distribution"
[image3]: ./images/orig.png "original image"
[image4]: ./images/shadow.png "shadow"
[image5]: ./images/bright.png "brightness"
[image6]: ./images/recover.png "recover"
[image7]: ./images/track2.png "track2"
[image8]: ./images/crop.png "crop"
[image9]: ./images/track1_sim.png "track1_sim"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md as the writeup to summarize the results
* [Track1 Video](https://youtu.be/O2CO8ZxgkFM)
![alt text][image9]
* Experiment model09.h5 work-in-progress for track2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for loading training data, augmenting data and saving the convolution neural network.
The file shows the pipeline and contains comments to explain how it works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use  [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf):

![alt text][image1], which use convolution neural network, and RELU layers for non-linearity.
Data is normalized in range \[-1.0, 1.0]


#### 2. Attempts to reduce overfitting in the model
I spilt data into training set and validation set to reduce overfitting effect.
I add weight regularization to convolution layers to reduce overfitting.

I didn't user dropout in my model, because it result in worse validation accuracy and bad performance in simulation.
I compare the results from batch normalization (BN) only, BN + dropout, and dropout. It turns out BN only has the best accuracy


#### 3. Model parameter tuning
I use Adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I used a combination of:
* Center lane driving.
* Recovering from the left and right sides to the center.
* Counter-clock-wise direction.
* Scenes which the model need more data points (bridge, sharp turns).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

For this project, I need a model to retrieving vision information from images so CNN is the primary choice.

My first attempt is [AlexNet](https://en.wikipedia.org/wiki/AlexNet). However, the trained model does not perform well in simulation.
The car keeps slow down for every 2 seconds for unknown reasons (even the validation error is pretty low).
After days of unsuccessful trails I decide to give up.

The next network I tried is [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). However, the network is too large and cause tensorflow out of memory
crash during training. I cannot fix this even reduce the batch size and other apporaches so I give up.

The eventual network I used is nVidia model. The paper does not mention what kinds of activation function to use, I choose to use RELU given its good performance in CNN.

After decide the network, I start to collect training data. I use the simulator in training mode to collect more than 10k images, with various scenarios.
Then I use the collected data to train the model and tune parameters.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.
To improve that, I collect those special scenarios only and add to the training data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
[Track1 Video](https://youtu.be/O2CO8ZxgkFM)

#### 2. Final Model Architecture
I start the network from nVidia model, with some modifications:

* I added batch normalization between each layers, which improves validation accuracy a lot.

* I also added weight regularization in CNN to further reduce overfitting.

* I tried to add dropout in various ways, including: between each fully-connected layers, inside CNN, BN + dropout, dropout only,
but not of them helps to reduce overfitting so I did not use droptout in my network. 

The final model architecture consisted of a convolution neural network and then fully connected layers:

* Input: 160x320x3
* Convolution 2D, kernel: 5x5x24, stride: 2
* Batch normalization + RELU
* Convolution 2D, kernel: 5x5x36, stride: 2
* Batch normalization + RELU
* Convolution 2D, kernel: 5x5x48, stride: 2
* Batch normalization + RELU
* Convolution 2D, kernel: 3x3x64, stride: 1
* Batch normalization + RELU
* Convolution 2D, kernel: 3x3x64, stride: 1
* Batch normalization + RELU
* Fully connect: 8448 * 100
* Batch normalization + RELU
* Fully connect: 100 * 50
* Batch normalization + RELU
* Fully connect: 50 * 10
* Batch normalization + RELU
* Fully connect: 10 * 1


#### 3. Creation of the Training Set & Training Process

#####Data Collection
To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to 
pivot back the center when the car is out of track.

![alt text][image6]

The above is also done in reverse lap direction (Counter-clock-wise) in order to collect more data points.

I tried the same process on track two. However, I didn't use such data finally because it cause the model performs worse in track1 (regression).

![alt text][image7]

#####Data Augment
The data distribution shows that steering=0.0 images is much more then others, and data is skewed leftward (too much left turns).
![alt text][image2]

So I augment images by randomly flipping them with probability 0.5, which would makes the data is more symmetric.

In order to avoid overfitting, I also augment images with random shadow and brightness. See the examples:

* original image

![alt text][image3]
* random shadow

![alt text][image4]

* random brightness

![alt text][image5]

#####Image Preprocess
For each image, I crop the image and exclude the upper 50 pixels and bottom 20 pixels. Since those pixels does not affect
the steering.

![alt text][image8]


#####Training
After the collection process, I had 20465 (8036 from Udacity repo) number of data points. 
I use 80% as  training data and 20% as validation data.
I use 10 as epochs, but also with checkpoint to output model form tensorflow in each epoch so I can pick the one with least validation error.

#### 4. Future works
THere are some works can be improved but I don't have enough to finish by now:
1. Pass track 2, need to collect more data points.
2. Try more data augmentation techniques, like translation, rotation to handle more complex scenes.
