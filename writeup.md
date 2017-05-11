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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


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

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
