# CS231n CNN for Visual Recognition

[L1/L2](#nearest-neighbor-classifier)  
[Hyperparameter Search](#validation-sets-for-hyperparameter-tuning)  
[Cross-validation](#cross-validation)  

## Module 1: Neural Networks
(L1/L2 Distances, hyperparameter search, cross-validation)
### Image Classification
Challenges: 
- **Viewpoint variation:** An object can be oriented in many ways with respect to the camera.
- **Scale variation:** Visual classes often exhibit variation in their size (size in the real world, not only in terms of their extent in the image)
- **Deformation:** Many objects of interest are not rigid bodies and can be deformed in extreme ways.
- **Occlusion:** The objects of interest can be occluded. Sometimes only a small portion of an object (as little as few pixels could be visible)
- **Illumination conditions:** The effects of illumination are drastic on the pixel level.
- **Background clutter:** The objects of interest may blend into their environment, making them hard to identify.
- **Intra-class variation:** The classes of interest can often be relatively broad, such as chair. Thera are many different types of these objects, each with their own appearance.

*The image classification pipeline.*
- **Input:** Our input is a set of N images, each labeled with one of K different classes. We refer to this data as the *training set*.
- **Learning:** Our task is to use the training set to learn what every one of the classes looks like. We refer to this step as *training a classifier* or *learning a model*.
- **Evaluation:** We evaluate the quality, by asking it to predict labels for a new set of images that it has never seen before.

#### Nearest Neighbor Classifier
Has nothing to do with CNN and is very rarely used in practice. CIFAR-10, one popular toy image classification dataset with 60,000 tiny images that are 32 pixels high and wide. Through a test, we can see that only 3 from the 10 images match their respective class. E.g. The nearest training image to the horse head is a red car, presumably due to the strong black background. As a result, this image of a horse would in this case be mislabeled as a car.

The way in that we compare two images which are blocks of 32 x 32 x 3 is through the L1 distance.
$$d_1(I_1,I_2) = \sum_p \| I_1^p - I_2^p \|$$

When comparing two images with L1 distance (for one color channel for example), two images are substracted elementwise and then all differences are added up to a single number. if two images are identical the result will be zero. But if the images are very differnet the result will be large.

#### How can we implement it?
- Xtr: Images for training
- Ytr: Labels for training
- Xte: Images for testing
- Yte: Labels for testing

```py
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/')
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
```

```py
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
     # using the L1 distance 
     distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
     min_index = np.argmin(distances) # get the index with smallest distance
     Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
```

```py
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
```

Running the code we see that the classifier only achieves **38.6%** on CIFAR-10, which is more impressive than guessing at random (which would give 10% of accuracy since there are 10 classes), but nowhere near human performance which is estimated at about 94%. In Kaggle we can see CNN's that achieve around 95% matching human accuracy.

Another common choice could be to instead use the L2 distance which has the geometric interpretation of computing the euclidean distance between two vectors.

$$d_2(I_1, I_2) = \sqrt{\sum_p(I_1^p - I_2^p)^2}$$

For the script we just need to change the L1 distances line for the L2 one.
```py
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```
If you ran it with this distance you would obtain **35.4%** accuracy.

#### K-Nearest Neighbor Classifier
You may have noticed that it is strange to only use the label of the nearest image when we wish to make a prediction.  
> I already know how this works in depth so im skipping the explanation

In practice, you will almost always want to use k-Nearest Neighbor. But what value of k should you use? We turn to this problem next.

Disadvantages:
- The classifier must remember all the training data and store it for future comparisons with the test data. This is space inefficient.
- Classifying a test image is expensive since it requires a comparison with all the training images.

#### Validation sets for Hyperparameter tuning

The k-nearest neighbor classifier requires a setting for k. But what number works best? Additionaly we saw that there are many distance functions. These choices are hyperparameters and they come up very often in the design of many ML algorithms that learn from data. 

You might be tempted to suggest that we should try out many different values and see what works best... right?

Thats what we are going to do, but it must be done very carefully. We CANNOT use the test set for the purpose of tweaking hyperparameters. Whenever we design ML algorithms, we should think of the test set as a very precious resource that should ideally never be touched until one time at the very end. Otherwise, the very real danger is that you may tune your hyperparameters to work well with the test set. In practice, we would say that we overfit to the test set. Another way of looking at it, is that you tune your hyperparameters on the test set, you are effectively using the test set as the training set.

> *Evaluate on the test set only a single time, at the very end.*

Luckily, there is a correct way of tuning the hyperparameters and it does not touch the test set at all. The idea is to split our training set in two: a smaller training set, and what we call a validation set. Using CIFAR-10 as example, we could for example use 49,000 of the training images for training, and leave 1,000 aside for validation. This validation set is essentially used as a fake test to tune the hyperparameters.

```py
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```

By the end of this procedure, we could plot a graph that shows which values of k work best. We would then stick with this value and evaluate once on the actual test set.

#### Cross Validation
**Cross validation**. If our training data is small, we can use a technique for hyperparameter tuning called cross validation. The idea is that instead of arbitrarily picking the first 1000 datapoints to be the validation set and the rest training set, we iterate over different validation sets and averaging the performance across these. For example in 5-fold-cross-validation, we would split the training data into 5 equal folds, use 4 of them for training, and 1 for validation. 

In practice, people prefer to avoid cross-validation cause is computationally expensive. The splits people tend to use is between 50%-90% of the training data for training and rest for validation.   
For example if the number of hyperparam. is large, you may prefer to use bigger validation splits. If the number of examples in the validation set is small (a hundred or so), it is safer to use cross-validation.

#

[Domingos, P. (October, 2012). A Few Useful Things to Know About Machine Learning, Vol.55 on 21/07/2023.](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)

The three components of learning algorithms
#### Representation
- Instances
     - K-nearest neighbor
     - Support vector machines
- Hyperplanes
     - Naive Bayes
     - Logistic Regression
- Decision Trees
- Sets of rules
     - Propositional rules
     - Logic programs
- Neural networks
- Graphical models
     - Bayesian networks
     - Conditional random fields

#### Evaluation
- Accuracy/Error rate
- Precision and recall
- Squared error
- Likelihood
- Posterior probability
- Information gain
- K-L divergence
- Cost/Utility
- Margin

#### Optimization
- Combinatorial optimization
     - Greedy Search
     - Beam Search
     - Branch-and-bound
- Continuous optimization
     - Unconstrained
          - Gradient descent
          - Conjugate gradient
          - Quasi-Newton methods
     - Constrained
          - Linear programming
          - Quadratic programming
          
#