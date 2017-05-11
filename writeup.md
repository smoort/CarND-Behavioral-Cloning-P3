# **Behavioral Cloning Project Write-up** 

### This is a write up on the Behavioural Cloning project


---

### **Goals of the Behavioral Cloning Project**


* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NormalImg.jpg "Normal Training"
[image2]: ./examples/RecoveryImg.jpg "Recovery Image 1"
[image3]: ./examples/RecoveryImg2.jpg "Recovery Image 2"
[image4]: ./examples/ActualImg.jpg "Normal Image"
[image5]: ./examples/FlippedImg.jpg "Flipped Image"


### *Model Architecture and Training Strategy*


**1. Model Architecture** : A convolutional neural network based on Nvidia architecture has been used for this solution.

* 5 convolutioanl layers have been used with increasing number of filters and RELU activation at each layer to introduce nonlinearity.
* 5 fully connected layers (including the output layer) have been used with decreasing number of nodes
* A Dropout has been implemented after the convolutional layers to reduce overfitting
* Input image cropping and normalization are handled within the model


**2. Attempts to reduce overfitting in the model**

* The model contains a dropout layer in order to reduce overfitting. 
* The model was trained and validated on different data sets to identify any overfitting situation.
* The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. This confirmed that the model was appropriately fit.


**3. Model parameter tuning**

* The model used an adam optimizer, so the learning rate was not tuned manually.


**4. Appropriate training data** : Multiple recordings were done as listed below to successfully train the model.

* Centre lane driving to provide the base data
* Zig zag driving to help the model come back to center when it drifts to the edges
* Additional training on spots where the model had challenges staying on road

---


### *Architecture and Training Documentation*


**1. Solution Design Approach**

The strategy used for deriving a model architecture is as listed below :

* A convolutional neural network was used to train a model that will determine the best steering angle for a given image
* Images were normalized and cropped to make training effecient
* Normal driving data was collected and initial training done.  Steering angle of 0 was ignored from this dataset to avoid the model getting biased towards straight driving.
* Zig zag driving provided key data for the model to learn how to reach when the car drifts towards the edges of the road
* Challenging spots were identified from each of the runs and more data was collected on those spots
* Keeping the epochs to 3 was found to be the most effective training - this kept the training time low and gave best results
* Generator with batch size 32 was used for training and validation to keep the memory requirements low.
* Data was augumented by flipping existing images to provide more test data.
* The weights were saved after every training so that they can be reused as the starting point for next training.  This increases the effeciency of the training.
* The training and validation losses were reviewed after every training to look for overfitting.


**2. Final Model Architecture**

Below is the final architecture of the model

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Image Cropping2D 		| 160x320x3 --> 75x320x3						|
| Normalization	Lambda  | (x / 255) - 0.5								|
| Layer 1 - Conv 5x5  	| 2x2 stride, 24 filters, RELU activation	 	|
| Layer 2 - Conv 5x5  	| 2x2 stride, 36 filters, RELU activation	 	|
| Layer 3 - Conv 5x5  	| 1x1 stride, 48 filters, RELU activation	 	|
| Layer 4 - Conv 3x3  	| 1x1 stride, 64 filters, RELU activation	 	|
| Layer 5 - Conv 3x3  	| 1x1 stride, 64 filters, RELU activation	 	|
| Dropout Layer    		| 0.25	 										|
| Fully connected 0		| Flatten										|
| Fully connected 1		| 1164											|
| Fully connected 2		| 100											|
| Fully connected 3		| 50											|
| Fully connected 4		| 10											|
| Output layer			| 1												|


**3. Creation of the Training Set & Training Process**


* The model was trained iteratively using the results of the previous run to identify sections that had to be retrained
* The first run was normal driving which was used for the initial training.  Steering angle of 0 was ignored from this dataset to avoid the model getting biased towards straight driving.
* Zig zag driving was done to help the model to learn how to reach when the car drifts towards the edges of the road
* Challenging spots were identified from each of the runs and more data was collected on those spots
* Data was augumented by flipping existing images to provide more test data.
* Preprocessing was built into the model - Normalization and cropping were the preprocessing steps applied
* The weights were saved after every training so that they can be reused as the starting point for next training.  This increases the effeciency of the training.
* Data was shuffled once read to avoid predictability
* A validation dataset was extracted from the input data.  This data was used for validation to check if the model was over or under fitting.
* 3 epochs provided the best results for this exercise.
* Adam optimizer was used and hence  manually training the learning rate wasn't necessary.

Image from normal driving
![alt text][image1]

Images from recovery driving
![alt text][image2]
![alt text][image3]

Image before flipping
![alt text][image4]

Image after flipping
![alt text][image5]
---