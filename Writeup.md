
#**Traffic Sign Recognition** 

##Writeup

---

**Build a Traffic Sign Recognition Project**

The goals/steps of this project are the following:
* Load the data set (see below for links to the project dataset)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/r0busta/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. A basic summary of the data set.

I used python to calculate summary statistics of the traffic
signs dataset:

* The size of the training set is 34.799 images
* The size of the validation set is 4.410 images
* The size of the test set is 12.630 images
* The shape of a traffic sign image is 32 px x 32 px (RGB)
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Data pre-processing

As a first step, I decided to convert the images to grayscale because shape and pictograms are more important than sign colors. Generally, traffic signs designed in a way that they can be recognized by colorblind people as good as by non-colorblind ones. That is why the color is not essential for the model training process.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As the last step, I normalized the image data so that data is distributed uniformly around zero. That speeds up the weights optimization process (i. e. gradient descent) and improves the training quality.

####2. Test data augmentation

After running a couple of training rounds, I found that the model was tending to overfit and prediction accuracy was always less than 89%. My assumption was that the model was relying too much on a sign orientation and position on the picture. So I decided to generate additional data.  

To add more data to the data set, I generated 5 extra images for an each image from the training set by randomly shifting and rotating pictures. Shifting was performed in both dimensions by [-2 px; +2 px] and the image was rotated by [-15 degree; +15 degree]. In total, extra 173.995 images were generated

Here is an example of an original image and an augmented image:

![alt text][image3]

####2. Model Architecture.

I've decided to take LeNet architecture. The final model following that architecture consists of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 Grayscale image                       | 
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x10   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 14x14x10   |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x16     |
| Fully connected       | Input 400, output 120                         |
| RELU                  |                                               |
| Fully connected       | Input 120, output 84                          |
| RELU                  |                                               |
| Fully connected       | Input 84, output 43                           |
| Softmax               |                                               |
 

####3. Model Training. 

To train the model, I used Adam optimizer and cross-entropy as a loss function. Loss function was calculated as a mean of the cross-entropy function of softmax logits and one-hot labels.

Other hyperparameters are number of epochs, batch size, and learning rate are chosen to be 10, 256 and 0.001 consequently.

####4. Solution Approach

To approach the solution for achieving the targeted accuracy of 0.93 on the validation set I decided to start from taking original LeNet architecture. I believe that this architecture suits the project problem very well because the model was designed for MNIST which consists of handwritten characters. These characters represent non-complex shapes as well as grayscaled traffic sign images. So my assumption is that a model designed for classifying handwritten characters can classify traffic signs with the same accuracy.

Giving it a try on the original train set of German traffic signs, the trained model showed a good performance of 0.89 accuracy on the validation set. After I augmented train data with generated data, the re-trained model showed the targeted accuracy of 0.93

My final model results were:
* training set accuracy of 0.99
* validation set accuracy of 0.94
* test set accuracy of 0.93

###Test a Model on New Images

####1. Arbitrary German traffic sign images

Here are five German traffic signs (resized to 32x32x3 shape) that I found on the web:

![alt text][./images/img_1.png] ![alt text][./images/img_2.png] ![alt text][./images/img_3.png] 
![alt text][./images/img_4.png] ![alt text][./images/img_5.png]

Some images might be difficult to classify:
* the first image - the sign is covered with show
* the fifth image - the sign is augmented by a street-art:)

####2. Model's prediction results

Here are the results of the prediction:

| Image                 |     Prediction           | 
|:---------------------:|:------------------------:| 
| General caution       | General caution          | 
| Slippery road         | Slippery road            |
| Speed limit (70km/h)  | Speed limit (70km/h)              |
| Children crossing     | Right of way at next intersection |
| No entry              | No entry                 |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.91

----------
WIP below

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
The softmax distribution shown on the following figure:

![alt text][softmax_plot.jpg]

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five softmax probabilities were

For the second image ... 

###Visualizing the Neural Network
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][layer_viz.jpg]
