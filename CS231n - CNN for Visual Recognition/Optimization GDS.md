# CS231n CNN for Visual Recognition

[Optimization landscape](#optimization-and-stochastic-gradient-descent)  
[Local Search](#random-local-search)  
[Analytical/Numerical gradient](#computing-the-gradient)  

## Module 1: Neural Networks
(Optimization landscapes, Local search, Learning-rate, Analytic/Numerical gradient)

# Optimization and Stochastic Gradient Descent

## Introduction

We previously introduced a **score function** that mapped the *image pixels to class scores*, and a **loss function** that measured the quality of a particular set of parameters based on how much the induced scores agreed with the ground truth labels in the training data.

Our linear function had the form of $f(x_i, W) = Wx_i$ and the SVM we developed was formulated as:
$$L = \frac{1}{N} \sum_i \sum_{j\not= y}[ max(0, f(x_i;W)_j - f(x_i ; W)_{y_i} + 1)] + aR(W)$$

We now need to find the set of parameters W that minimize the loss function, thats what's called **optimization**.

## Visualizing the Loss function

In CIFAR-10 a linear classifier weight matrix is size [ 10 x 3073 ] for a total of 30,730 parameters, that's difficult to visualize. However, we can still gain some intuition through 1 or 2 dimentional spaces because they are visualizable. We can generate a random weight matrix and then move in a direction while recording the loss function values along the way.   For one-dim we can do $L(W + \alpha W_1)$ for different values of $\alpha$. That will create a simple plot with the value of $\alpha$ in the X-axis and the loss in the Y-axis. 

![1-dim plot](https://cs231n.github.io/assets/svm1d.png)

For 2 dimentions we just evaluate the loss in $L(W + aW_1 + bW_2)$ as we vary a, b. In the plot a,b is the X-axis and the Y-axis, and the value of the loss function is the color.

![2-dim plot ex1](https://cs231n.github.io/assets/svm_one.jpg)![2-dim plot ex2](https://cs231n.github.io/assets/svm_all.jpg)


We have:
$$L_i = \sum_{j\not=y_i} [max(0, w_j^T x_i - w_{y_i}^T x_i + 1)]$$
which represents the loss for a single example i. We sum every class j except the correct class y_i. Then in the operations, its calculating the difference between the prediction and the ground truth, then doing a ReLU which introduces non-linearity by thresholding the output.

Then the SVM loss for each class $(L_0, L_1 and L_2)$ is computed as the sum of the loss for each example and then averaged. 

$$L_0 = max(0, w_1^T x_0 - w_0^T x_0 + 1) + max(0, w_2^T x_0 - w_0^Tx_0 + 1)$$
$$L_1 = max(0, w_0^T x_1 - w_1^T x_1 + 1) + max(0, w_2^T x_1 - w_1^Tx_1 + 1)$$
$$L_2 = max(0, w_0^T x_2 - w_2^T x_2 + 1) + max(0, w_1^T x_2 - w_2^Tx_2 + 1)$$
$$L = (L_0 + L_1 + L_2)/3$$

![](https://cs231n.github.io/assets/svmbowl.png)

This is a 1-dim illustration of the loss, the full SVM data loss is a 30730 dimensional version of that shape.

## Optimization

The goal is to find W that minimizes the loss function. We will slowly develop an approach to optimize the loss function.

### Random Search

A very bad idea but that is very simple. Let's try out many different random weights and keep track of what works best.
```py
best_loss = float("inf")
for num in range(1000):
    W = np.random.randn(10, 3073) * 0.00001
    loss = Loss(X, y, W)
    # change best_loss if loss is smaller
```

After running this, it gives an accuracy of **15.5%**, random its 10%.

> Our strategy now will change, instead of iterating through random weights we will initialize them and then refine them slowly to minimize the loss.

You can see the problem as the `blindfolded hiker analogy`. Imagine hiking at a hilly terrain with a blindfold on and trying to reach the bottom. In the example of CIFAR-10 the terrain would be 30730 dimentional where at every point of the hill we achieve a particular loss.

### Random Local Search

Continuing the blindfolded hiker analogy, one idea would be to try taking a step in some random direction, if it goes down continue doing steps, repeating this process until reaching the bottom. 

So we will have our initial $W$ and then generate random perturbations $\delta W$ and if the loss at W + $\delta W$ is lower we will perform an update. 

```py
W = np.random.randn(10, 3073) * 0.001 
bestloss = float("inf")
for i in range(1000):
  step_size = 0.0001
  Wtry = W + np.random.randn(10, 3073) * step_size
  loss = L(Xtr_cols, Ytr, Wtry)
  # change best_loss if loss is smaller
```
Doing 1000 iterations as well as before, it reaches an accuracy of **21.4%**. This is better but still wasteful and computationally expensive.

### Following the Gradient

Thanks to the beauty of math, we don't need to search for a good direction, we can compute the best direction and it will be mathematically guaranteed to be the direction of the steepest descent. This is done through the gradient of our loss function.

We already know what a gradient is and how to compute it so I will not enter in details.

## Computing the gradient

There are two ways to compute the gradient: A) slow but best way (numerical gradient) and B) a fast exact but prone to error way that requires calculus (analytic gradient).

### Numerically with finite differences

It iterates through each dimension making a small change along that dimension and calculating the partial derivative of it. 

**Practical considerations**. Note that in the mathematical formulation the gradient is defined in the limit as h goes towards zero, but in practice it is often sufficient to use a very small value (such as 1e-5 as seen in the example). Ideally, you want to use the smallest step size that does not lead to numerical issues. Additionally, in practice it often works better to compute the numeric gradient using the **centered difference formula**: 
$$\frac{f(x+h)−f(x−h)}{2h}$$

**A problem of efficiency**: Evaluating the numerical gradient can be very expensive, in our example we had 30730 parameters in total and had to do 30731 evaluations of the loss function to evaluate the gradient and perform a single parameter update. Modern Neural Networks have millions of parameters so we probably need some better strategy.

### Analytically with calculus

Through calculus we can get a direct formula for the gradient (no approximations) that is also very fast to compute. The drawback is that its more error prone to implement which is why in practice its common to compute both gradients to check if your implementation is correct, this is known as a **gradient check**.

## Gradient Descent

Now that we've computed the gradients, the procedure of constantly evaluating it and performing a parameter update is called gradient descent.

This one is currently and by far the most common and established way of optimizing Neural Networks loss functions. 

### Mini-batch Gradient Descent

Training data can have millions of samples in big datasets which nowadays are more common than ever. Hence, its seems wasteful to compute the full loss function over the entire training set just to perform a single parameter update. A very common approach is to compute the gradient over batches, currently, a typical batch contains 256 examples from a training set of 1 million.

The gradient from a minibatch is a good aproximation of the gradient by its full objective but much faster convergence can be achieved with it.

The extreme mini-batch case is called **Stochastic Gradient Descent**. This is less common to see due to vectorized code optimizations, it can be much more efficient to compute the gradient for 100 examples than 100 times the gradient for a single example. The batch size is a hyperparameter that usually is attached to the memory constraints. We usually use power of two values because many vectorized operations implementations work faster when their inputs are powers of two.