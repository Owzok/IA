# CS231n CNN for Visual Recognition

[Parametric approach]()
[Bias trick]()
[Hinge loss]()
[Cross-entropy loss]()
[L2 regularization]()
[Web demo]()

## Module 1: Neural Networks

### Linear Classification
Now we are going to develop a **more powerful image classifier** that will lead us right onto NN and CNN's. We will now have a score function that maps raw data to class scores and a loss function which quantifies the agreement between the predicted scores and the ground truth labels.

### Parameterized mapping from images to label scores

In CIFAR-10 we have a training set of $N = 50000$ images each with a D (dimension) of $32x32x3 = 3072$ pixels and K (classes) $= 10$.

Our linear classifier, the simplier linear mapping.

$$f(i_1, W, b) = W_{x_i} + b$$

| Parameter | Description | Shape | 
| --- | --- | --- |
| xi | Pixels of ith image flattened onto a vector | 3072 x 1 |
| W | Weights matrix | 10 x 3072 |
| b | bias vector | 1 x 10 |
| Wxi | Class score *(pre bias)* | 10 x 1 

![cat](https://cs231n.github.io/assets/imagemap.jpg)

We could see images as high-dimensional points. We can interpret each image as a point in a 3072-dimensional space. Analogously, the entire dataset is a labelet set of points.

Since we defined the score of each class as a weighted sum of all image pixels, each class score is a linear function over this space. We cannot visualize a 3072-dimensional space but we can squash it and try to visualize the idea in 2 dimensions.

![linear](https://cs231n.github.io/assets/pixelspace.jpeg)

#### Bias trick
A common simplifying trick to represent w, b as one single variable.
The trick consists on combining W and b into a single matrix that holds them both and extending xi by a dimension that always contain a constant 1, with that, the new score function would look like:
$$f(x_i, W) = Wx_i$$

![bias trick](https://cs231n.github.io/assets/wb.jpeg)

In machine learning the features are always normalized. In the examples above we have been treating the pixels in their RGB range (0-255). First we need to center the data by substracting the mean from every feature, that would give us a range between -127 and 127 and after normalization, -1 to 1.

#### Loss Function
We have defined a function from pixel values to class scores, parametrized by W. Our data (xi and yi) is fixed  and given, so it cannot be modified, but we do have control over these weights and we want to set them so our predicted classes are consistent with the ground truth labels.

We are going to measure our unhappiness with outcomes through a **loss function** (sometimes referred to as the **cost function** or **objective**). The loss will be high if we are doing a poor job of classifying data, and low if we are doing well.

#### Multiclass SVM loss
The SVM loss is set up so that the SVM wants the correct class to have a higher score than the incorrect classes by a delta margin.

Recall that for the i-th example we are given pixels for the xi image and yi being the label. The score function takes xi and computes the vector f(xi, W) which we will abreviate as s. Then the multiclass svm loss for the i-th example is then formalized as follows:
$$L_i = \sum_{j \not = y_i}max(0, s_j - s_y + \Delta)$$

**Example:** Suppose we have three classes which received the scores s = [13, -7, 11], the first one is the tru class (i.e. $y_i = 0$). Also assume that $\Delta$ is 10.
$$L_i = max(0, -7 - 13 + 10) + max(0, -11 - 13 + 10)$$
You can see the first term gives 0 since -7 - 13 + 10 gives a negative number, which is then trhresholded to zero with the **max(0, -)** function. We get zero loss for this pair because the correct class score 13 was greater than the incorrect class score (-7) by a least margin of 10. The second term computes [11 - 13 + 10] which gives 8, even the correct class was greater than the the incorrect class it wasn't greater than the desired margin of 10.

We are working with linear score functions so we can rewrite the loss function as follows:
$$L_i = \sum_{j=y_i}max(0,x_j^T x_i - w_{y_i}^T x_i + \Delta)$$

$L_i$: This is the scalar loss for the i-th example.

$j$: This is an index that ranges from 0 to the number of classes minus 1. It is used in the summation to loop over classes.

$y_i$: This is the true class label for the i-th example. It indicates which class is the correct class for this example. It's a scalar.

$\max(0, x_j^T x_i - w_{y_i}^T x_i + \Delta)$: This term represents the loss for the j-th class in the i-th example. Let's break down the size of each element within this term:

$x_j$: This is a feature vector for the j-th class. It's typically a column vector of size (D, 1), where D is the dimensionality of the feature space.

$x_i$: This is a feature vector for the i-th example. It's also typically a column vector of size (D, 1), where D is the dimensionality of the feature space.

$w_{y_i}$: This is a weight vector associated with the true class label $y_i$. It's typically a column vector of size (D, 1), where D is the dimensionality of the feature space.

$\Delta$: This is a margin hyperparameter and is just a scalar value.

So, in summary, the elements involved in the loss function have the following sizes:

Scalars: $L_i$, $y_i$, $\Delta$
Vectors of size (D, 1): $x_j$, $x_i$, $w_{y_i}$

The threshold at $max(0, -)$ function is called **hinge loss**. We'll sometimes hear people talk about **squared hinge loss** SVM which uses $max(0,-)^2$ that penalizes violated margins strongly.

#### Regularization
Suppose that we have a *dataset* with a set of parameters **W** that correctly classifies every example. The issue is that this **W** may not be unique, one easy way to see this, is that if a **W** correctly classifies every example then $\lambda W$ should also work for $\lambda \gt 1$.

In other words, we wish to encode some preference for some for a certain set of weights over others to remove this ambiguity. We can do this by extending the loss function with a **Regularization Penalty R(W)**. The most common one L2 looks like this
$$R(w) = \sum_k \sum_l w^2_{k,l}$$

Then the full Multiclass SVM loss becomes:

$$L = \frac{1}{N} \sum_i L_i + \lambda R(W)$$

Or expanding it onto its full form:

$$L = \frac{1}{N} \sum_i \sum_{j\not = y_i}[max(0, f(x_i; W)_j -f(x_i; W)_{y_i} + \Delta)] + \lambda \sum_k \sum_l W_{k,l}^2$$

Where there is no way to simple way of getting $\lambda$, its usually found through cross-validation.

Note that biases do not have the same effect since unlike the weights, they do not control the strength of influence of an input dimension. 

Loss function without regularization
```py
def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  correct_class_score = scores[y]
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0
  for j in range(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i

def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i

def L(X, y, W):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  # evaluate loss over all examples in X without using any for loops
  # left as exercise to reader in the assignment
```

#### Setting Delta.
What value should we assign to it? Should we find it through cross-validation? It turns out that it can safely be 1.0 in all cases. The $\Delta$ and $\lambda$ could look different but they control the same tradeoff between the data loss and regularization loss.

### Softmax Classifier

The other popular option is this one. Unlike the SVM which treats the outputs f(x_i, W) as uncalibrated and possibly difficult to understand scores for each class, this one gives a more intuitive input, normalized class probabilities. We now replace the **hinge loss** for **cross entropy loss** that has the form:

$$L_i = -log(\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}) \text{ or equivalently } L_i = -f_y + log \sum_j e^{f_j}$$ 

where $f_j$ is the scores f of the j-th element. The function $f_j(z) = \frac{e^{z_j}}{\sum_k e^z_k}$ is the softmax function, it takes a vector of real-values in z and squashes them into values from zero to one. 

The cross entropy between a true distribution p and an estimated distribution q is defined as:
$$H(p,q) = - \sum_x p(x)log\space q(x)$$

## SVM vs Softmax
![a](https://cs231n.github.io/assets/svmvssoftmax.png)

SVM gives more difficult to understand scores while softmax gives normalized "probabilities", between quotes because they are not real probabilities. $\lambda$ is going to affect a lot on the probabilities because of the regularization. 

In practice both are comparable and a lot of people will have different opinions on which one works better. 

The softmax will never be happy with the results it gets, since the correct class will always have a higher chance than the incorrect one so thw loss would always get better. However, the SVM is happy once the margins are satisfied and it doesn't manage perfect scores after that.

### Readings
[Deep Learning using Linear Support Vector Machines](https://arxiv.org/abs/1306.0239) from Charlie Tang 2013 presents some results claiming that the L2SVM outperforms Softmax.