## Yes you should understand backprop
[Karpathy Medium](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)  
Andrej Karpathy, Dec 19, 2016

> "The problem with Backpropagation is that it is a leaky abstraction."  

A **leaky abstraction** refers to a situation where the details of the underlying process are hidden or abstracted away, but those details still have a significant impact on the overall system behaviour. Backpropagation hides complexities and challenges that can arise during the training process.

If the **weights initialization** is badly done, it can affect the whole training process. When the sigmoid function is too large or too small the output from the sigmoid becomes too close with 0 and 1. In that range, the gradient becomes very close to 0. That is called, ```saturation of Non-linearities```.

When Andrej refers to "Vanishing Gradients" it refers to a problem when training within the tanh function. When **gradients become very small**, they can cause the weights to **update very little** or not at all.

Lets check this example:
```py
a = -np.dot(W,x)            # Input * Weight
z = 1/(1 + np.exp(a))       # sigmoid, forward pass
dx = np.dot(W.T, z*(1-z))   # backward pass: local gradient for x
dW = np.outer(z*(1-z), x)   # backward pass: local gradient for W
```

### Reminding
The forward pass, is the process to go from the input through every layer until obtaining the output. In this case, the forward pass calculates the output ```z``` of the fully connected layer using the sigmoid activation function.
```W``` is the weights matrix of the connected layer and ```x``` is the vector of input data. 

The backward pass is the process of calculating the gradients of the loss, which is essential for updating the parameters during training. 
```z*(1-z)``` is the derivative of the sigmoid function. Doing a dot product with the Weights transposed, results on the local gradient for X.

The other backward pass now with W instead of x, is the outer product for the derivative of the sigmoid and the input data.

### Back to the problem

As Andrej mentioned above, if the weights matrix ```W``` is initialized with large values it can cause the output ```np.dot(W, x)``` to have a very large range. As a consequence, the sigmoid would return in almost binary outputs (almost all being 0 or 1). After that, the derivative ```z(1-z)``` becomes very close to zero, resulting in the local gradients multiplying with a very small number and making them disappear. 

To address this issue, many people now is using a more robust and non-saturated function, the ReLU function instead of the popular sigmoid/tanh. ReLU have helped mitigate the vanishing gradient problem and have become more prevalent in modern architectures.

Another non-obvious fun fact about sigmoid is that its local gradient achieves a maximum of ```0.25```, when ```z = 0.5```.

### Dying ReLU's
Another fun non-linearity is the ReLU, which thresholds neurons at zero from below. 
```py
z = np.maximum(0, np.dot(W, x)) # forward pass
dW = np.outer(z > 0, x) # backward pass: local gradient for W
```
The problem appears when a ReLU neuron gets *clamped to zero*, making them not to fire during the forward pass and the weights receive zero gradients during the backward pass. When the gradients are zero, neuron's weights won't be updated and the neuron will *remain dead for the future inputs*. This can occur if weights are initialized in such a way that it **always produces negative outputs**. 

To address this problem and keep the benefits from the ReLU, there are some variations like:
- Leaky ReLU: Instead of setting negative values to zero, it multiplies negative inputs by a **small positive slope** (e.g. 0.01) to allow a small gradient flow for negative values.
- Parametric ReLU (PReLU): Similar to the leaky relu but instead of using a fixed slope, the slope is treated as a learnable parameter during training, allowing the network to adaptively **determine the slope** for each reunion.

### Exploding Gradients in RNNs

Vanilla Recurrent Neural networks processes sequences of data by mantaining hidden state information accross time steps. 

> TODO: Research about RNN's

```py
H = 5 # dimensionality of the hidden state
T = 50 # number of time steps

whh = np.random.randn(H,H)

# forward pass of a RNN (ignoring outputs x)
hs = {}
ss = {}
hs[-1] = np.random.randn(H)
for t in xrange(T):
     ss[t] = np.dot(Whh, hs[t-1])
     hs[t] = np.maximum(0, ss[t])

# backward pass of the RNN
dhs = {}
dss = {}
dhs[T-1] = np.random.randn(H) # start off the chain with random gradient
for t in reversed(xrange(T)):
     dss[t] = (hs[t] > 0) * dhs[t] # backprop through the nonlinearity
     dhs[t-1] = np.dot(Whh.T, dss[t]) # backprop into previous hidden state
```PP

The issue here arises from the recurrent connection and matrix multiplication. 
```py
dhs[t-1] = np.dot(Whh.T, dss[t])
```
- **Vanishing gradients:** If the largest eigenvalue of the recurrence matrix ```Whh``` is less than one, the gradient signal will diminish exponentially as it propagates backward through time. This can lead to the weights barely updating, causing slow learning and difficulties in capturing long-range dependencies in the data.
- **Exploding gradients:** If the largest eigenvalue of Whh is greater than one, the gradient signal will grow exponentially, leading to extremely large gradients. This can cause unstable training and result in numerical issues like overflow.

To address the vanishing and exploding gradient problem in RNNs, one common approach is gradient clipping, where the gradients are scaled down if they exceed a certain threshold. Another approach is to use Long Short-Term Memory (LSTM) networks, which are designed to better handle long-range dependencies and mitigate the gradient-related issues.
