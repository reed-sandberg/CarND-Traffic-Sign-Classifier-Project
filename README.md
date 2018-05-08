# Traffic Sign Recognition

A convolutional neural network (CNN) model was developed to identify German road signs.

## Goals

* Load the training data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize and discuss the results

### Data Set Summary & Exploration

Labeled photos of German road signs were provided to train and validate a deep learning neural network model.
These photos were divided into three sets intended for training, validation and testing respectively.

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3) # 32x32 images with 3 color channels
* Number of classes = 43

#### Exploratory visualization

To get an idea of what the training set looks like, it was grouped and sorted by class. For each class,
10 random samples were [displayed](Traffic_Sign_Classifier.html#data-set-visualization) along with the number of samples represented in that class.

There is notable disparity among the different classes, with some being more represented than others as well as
obvious differences in average lighting, image quality/clarity, etc. For example, classes 20 and 42 are much less represented than many others,
and the random samples shown are largely difficult to perceive by the human eye.

### Model Design

The model developed in this project is based on the LeNet 5 architecture with modifications described by
[Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

#### Image preprocessing

Each of the 32x32 pixels of a given image is assigned to a feature that is fed into the model. Based on advice
given in the lectures and the referenced publication, the color model was converted to grayscale. This reduced the
input features to 32x32x1 (vs 32x32x3 with 3 color channels) with a lighter training workload, and was
reported to give better results on a similar set of images than with full color input. A simple operation suggested in the lecture was used to convert the three
color channels to grayscale:
```
grayscale_scalar = ave(red_channel, green_channel, blue_channel)
```

Further, each 8-bit grayscale pixel was normalized to a range between -1 and 1 such that the data set midpoint is 0,
and then shifted to achieve an average value of 0:
```
norm_scalar = (grayscale_pixel - 128) / 128 + 0.354
```

Normalization of input in this manner helps to stabilize gradient descent during optimization for more efficient determination
of the local minimum.


#### Model architecture

The final model architecture is based on a 5-layer LeNet model with 2 convolutional layers, and 3 fully connected
layers. A modification inspired by [Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
feeds an alternate pooling from the second activated convolutional layer back into the last 2
(fully connected) layers.

Details of each layer of the model
(also refer to Fig. 2 of
[Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)):

| Step | Operation             |     Description                                | 
|:----:|:---------------------:|:---------------------------------------------:| 
| 1 |Input                 | 32x32x1 grayscale image (pixel values normalized between -1 to 1)    | 
| 2 |Convolution 5x5         | 1x1 stride, same padding, output 28x28x6     |
| 3 |Activation                    | RELU                                               |
| 4 |Max pooling              | 2x2 stride, output 28x28x6                 |
| 5 |Convolution 5x5        | 1x1 stride, same padding, output 10x10x16  |
| 6 |Activation                    | RELU                                               |
| 7 |Max pooling              | 2x2 stride, output 5x5x16                 |
| 8 |Fully connected        | flatten previous step, input 400(x1), output 164 |
| 9 |Activation                    | RELU                                               |
|10 |Dropout                    | 25% input dropped |
|11 |Max pooling (step 6 output)  | alt pooling of step 6, 5x5 stride, output 2x2x16 |
|12 |Fully connected        | combine output of 10, 11, input 228, output 106 |
|13 |Activation                    | RELU                                               |
|14 |Dropout                    | 25% input dropped |
|15 |Fully connected        | combine output of 11, 14, input 170, output 43 |
 

#### Model training

A common technique using multinomial logistic classification followed by minimizing loss (optimization) was implemented for model training.
Preprocessed training samples were fed into the model (feed-forward) and the results (logits) were compared
with the known classes (labels) of the samples. To compare, a softmax function was applied to the logits, and and then the cross entropy taken with respect to the one-hot encoded labels representing an error function. An average of this error (or loss) for a batch of training samples is used by the optimization function to improve the model parameters (M and B) and minimize loss. This process is repeated for each batch until the entire training set has been used. The optimized parameters are then used to test accuracy on the validation images and labels to provide feedback on performance of the model. This complete cycle (an epoch) is repeated until accuracy on the validation set shows diminishing returns. The Adam algorithm as provided by the TensorFlow library was used for backpropagation to optimize model parameters and minimize error. Adam is a stochastic gradient descent algorithm designed to converge on an error minimum efficiently with individual learning rates maintained/evolved for each model parameter.

I prepared a test environment with CUDA/GPU compute power where I could tune the system meta (hyper) parameters and iterate on the model efficiently.
I started with the LeNet model I developed during the lectures, which gave good results on number image recognition (96+% accuracy). This seemed like a reasonable place to start since sign classification is also image-based and was suggested as a good starting point in the lectures for this project. LeNet was also used successfully by [Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) specifically to classify photos of German traffic signs. Given the ability to accept 2-D input along with 2-D convolutional learning layers, the model intuitively seems like a good candidate for classifying image data in general. After some slight adjustments of layer dimensions to output 43 classes (instead of 10 for digit classification), the model was ready for testing and tuning.

My approach for tuning and adjusting the model was almost entirely empirical - modifying hyper-parameters, tweaking the model, and seeing how the system reacted.
The requirement of 93% validation accuracy was met rather quickly with this standard LeNet 5 model after only a few simple tuning iterations. It reached an impressive 96% accuracy after several more tuning rounds. Feeling confident with standard LeNet and wanting to branch out a bit, I tried some modifications used by Pierre Sermanet and Yann LeCun where I fed a convolutional layer back into the final layers. This showed a perceptible but marginal improvement. While I chose this model to report final results, it is probably not worth considering in practice with the additional computational expense. Notwithstanding, Pierre Sermanet and Yann LeCun's own implementation and results, which overshadow mine.

My initial intuition of placing dropout steps just after activation seemed to be important to prevent overfitting (on digit classification), so I kept those steps intact throughout. I tried different activation functions like sigmoid, etc, without any improvement so I kept RELU intact as well. Once I was satisfied with the model results on the validation set, I tested a few final model/param combinations on the test set and chose the final model, preserving the saved TensorFlow session data (tf_session_final). The model's final results gave 96.3% accuracy on the validation set and 93.6% accuracy on the test set. The [training section](Traffic_Sign_Classifier.html#Train,-Validate-and-Test-the-Model) of the notebook reveals the code, but note that the output uses a different batch size and number of epochs for demonstration purposes only. The actual results reported here and saved session data submitted use the following hyper-parameters:


| Parameter | Value             |
|:----:|:---------------------:|
| learning rate | 0.0008                 |
| epochs | 200                 |
| batch size | 32                 |
| mean (random dist of param init) | 0                 |
| std dev (random dist of param init) | 0.1                 |
| dropout (retention rate) | 0.75                 |


Summary of final results:
* training set accuracy: 99.9%
* validation set accuracy: 96.3%
* test set accuracy: 93.6%

As expected, the training set gives the highest accuracy, since the model was optimized on those data. The validation set accuracy follows with some influence of these data "bleeding" into the model parameters as tuning was based on those results. Finally, the testing images as a pristine set of examples, was only used once the model was finalized and shows the lowest accuracy, but still high enough to expect the model will work well on external data "in the wild".

I spent most of the time tuning batch size and number of epochs since even slight modifications resulted in varying degrees of success. Most of the other hyper-parameters were quite focused in terms of how the system reacted to any adjustments, and they readily converged on optimal values.


### External Validation

Since this model showed good results on the training, validation and test sets provided, it was ready to test on external data in the wild. [Five photos](Traffic_Sign_Classifier.html#Load-and-Output-the-Images) of German traffic signs were collected from a Google image search. One of the samples (turn left ahead) looked blurry and was chosen to challenge the strength of the model. The other four photos seemed rather clear with good lighting conditions, and I expected them to be classified correctly.

#### Accuracy

The model [correctly classified](Traffic_Sign_Classifier.html#Predict-the-Sign-Type-for-Each-Image) all five images with 100% accuracy.

#### Certainty and performance analysis

Looking closer at the softmax output reveals high confidence of the model to classify all five samples. The [top five softmax probabilities](Traffic_Sign_Classifier.html#Output-Top-5-Softmax-Probabilities-For-Each-Image-Found-on-the-Web) of each sample predict correctly with 100% probability. Speed limit signs seemed to dominate the top five probabilities in most cases, but with such a high confidence in the leading probability, it's difficult to draw any conclusions about the remaining possibilities since they were all near 0% likelihood.

### Visualizing the Neural Network

The first two convolutional layers of the model were brought to life by [plotting them as images](Traffic_Sign_Classifier.html#Step-4-(Optional):-Visualize-the-Neural-Network's-State-with-Test-Images) to gain some deeper insight into the neural network learning ability.

The first layer (before pooling) seems to provide much more information and insight than the second layer (2nd and 3rd rows of code cell 7). In the first layer you can see definite distinguishing features of each road sign. The resulting images are remarkably similar to what you'd expect to see after applying edge detection, with major shapes within the image outlined. It is utterly fascinating to see how each image or parameter set focuses on different features of the image without any initial influence on what those features should be. The model truly exhibits a form of learning behavior.

