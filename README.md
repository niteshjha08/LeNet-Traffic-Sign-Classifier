# **Traffic Sign Recognition** 

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report  

---

Code used for this project is in this [Jupyter notebook](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data labels are distributed in the training set.

![distribution](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/distribution.PNG)

It is seen that speed limit labels 30,50,60,70,100,120 km/h have high occurences, whereas labels such as 'Road Narrows on right', 'Double curve' etc. have very few occurences. Thus we have a class imbalance and 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The images were converted using `(img-128)/128` to make them lie in the range [-1,1] as this significantly increased the speed of convergence and accuracy. For example, the first epoch gave a validation accuracy of only about 36% with original images, but using this 'standardization' (not exactly, as we don't use the mean to center it about 0), the validation accuracy reaches about 75% in the first epoch.

Here is an example of a traffic sign image before and after this conversion.

![original_img](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/original_img.PNG)   ![normalized_img](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/normalized_img.PNG)



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| ReLU		 			|         										|
| Maxpooling 2x2		| 2x2 stride, outputs 5x5x16      				|
| Fully connected		| outputs 300x1									|
| Fully connected		| outputs 120x1									|
| Fully connected		| outputs 43x1 									|

The number of nodes of fully connected layer was arrived at iteratively for best accuracy in validation set.
In the last fully connected layer, dropout was used (keep_prob=0.7) to avoid overfitting. 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an used a batch size of 128. A learning rate of 0.001 showed some fluctuations, so a rate of 0.0008 was chosen as it provided reasonably quick learning with minimal fluctuations in accuracy.
As the accuracy plateaus and remains the same at about 30 iterations. The training accuracy reaches 99% at this point. Higher epochs may result in slight increase in accuracy in validation set. I decided to keep it at 30 to avoid overfitting the data and stop when increase in accuracy of validation set is negligible.

I used the Adam Optimizer as it is faster than SGD and is commonly used for deep learning models.

Here is the learning curve of training data:
![training_acc](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/training_acc.PNG)

Here is the learning curve of validation data:
![training_acc](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/valid_acc.PNG)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 93.2% 
* test set accuracy of 93.38%

Initially, the original LeNet architecture was used with 120,84,43 nodes in the fully connected layers, and 6,16 filters used respectively in the convolution layers. However, this gave low accuracy. Hence, the number of nodes in fully connected layers was changed to 300,120,43. This improved the accuracy of training and validation set.
As the number of adjustable variables are large, I decided to keep this architecture constant and tried to tune the other hyperparameters. 

I arrived at the upper and lower bound of batch size to be about 100-200, i.e. accuracy would be highest between these values. For the learning rate, this came to be around 0.0005-0.001.
I tried combinations of 128,200 for batch sizes and 0.0005,0.001 and 0.008 learning rate and the results were such:

![valid_acc_agg](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/aggregated_valid_acc1.PNG)


We can see that the smaller batch gave more stable results, and 0.001 learning rate caused some erratic changes in accuracy.
Thus, I then tried it at learning rate of 0.0008 for both batch sizes.


![valid_acc_agg](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/aggregated_valid_acc2.PNG)


Although the accuracy in this run was lower for the 128 batch size, it was generally higher in the previous runs.

From these results, the final hyperparameters were chosen as:
* Fully connected layer nodes: 300-120-43
* batch-size: 128
* learning rate: 0.0008

As we don't want to overfit, dropout was applied on the final fully connected layer with keep_prob= 0.7.

### Test a Model on New Images

#### 1. I chose eight  traffic signs found on the web

Here are eight German traffic signs that I found on the web:

![new_images](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/new_images.PNG)


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![new_images_result1](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/new_images_result1.PNG)  ![new_images_result2](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/new_images_result2.PNG)

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No Passing   			| No passing									|
| Yield					| Yield											|
| 100 km/h	      		| 100 km/h						 				|
| No entry				| No entry		      							|
| Keep right			| Keep right									|
| 30 km/h				| 30 km/h										|
| Priority road			| Priority road									|


The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of which is 93.38%.
However, it did fail to give 100% accuracy in some runs, where one or two images were incorrectly classified. This may happen due to the images being one of the fewer occuring ones in the training set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is sure that this is a 30km/h limit sign, and the image does contain a 30km/h limit sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .992         			| 30 km/h    									| 
| .717     				| Road Work 									|
| .8 x e-03				| End of 80km/h									|
| .1 x e-03	      		| Go straight or right			 				|
| .1 x e-06			    | 70km/h		    							|

For the second image, the model is relative sure that this is a 100km/h limit sign, and the image does contain a 100km/h limit sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .739         			| 100 km/h    									| 
| .254     				| 80 km/h	 									|
| .0003					| 120 km/h										|
| .000229	      		| No Vehicles					 				|
| .00002			    | 30 km/h		    							|

For the third image, the model is completely sure that this is a Yield sign, and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Yield		   									| 
| 0     				| 20 km/h	 									|
| 0						| 30 km/h										|
| 0			      		| 50 km/h						 				|
| 0					    | 60 km/h		    							|

For the fourth image, the model is completely sure that this is a stop sign, and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Stop		   									| 
| 0     				| No entry	 									|
| 0						| Bicylces crossing								|
| 0			      		| Bumpy road					 				|
| 0					    | Road work		    							|

For the fifth image, the model is completely sure that this is a No-passing sign, and the image does contain a No-passing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| No passing   									| 
| 0     				| No passing for vehicles over 3.5 metric tons	|
| 0						| End of no passing								|
| 0			      		| Dangerous curve to the left	 				|
| 0					    | Slippery road	    							|

For the sixth image, the model is completely sure that this is a Keep right sign, and the image does contain a Keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Keep right   									| 
| 0     				| Turn left ahead								|
| 0						| 20 km/h										|
| 0			      		| 30 km/h						 				|
| 0					    | 50 km/h		    							|

For the seventh image, the model is completely sure that this is a No entry sign, and the image does contain a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| No entry		   								| 
| 0     				| No passing	 								|
| 0						| Stop											|
| 0			      		| End of all speed and passing limits 			|
| 0					    | Traffic signals	    						|

For the eight image, the model is sure that this is a Priority sign, and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999         		| Priority road									| 
| e-04     				| End of all speed and passing limits			|
| 0						| End of no passing								|
| 0			      		| Traffic signals				 				|
| 0					    | Right-of-way at the next intersection			|

Here is a visualization of the same result:


![barplot1](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/barplots1.PNG)  ![barplot2](https://github.com/niteshjha08/LeNet-Traffic-Sign-Classifier/blob/master/examples/barplots2.PNG)

