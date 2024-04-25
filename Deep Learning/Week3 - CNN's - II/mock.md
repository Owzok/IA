## Theory Section:

1. Define deep learning and explain its significance in modern artificial intelligence research. Discuss some key applications where deep learning has made significant contributions.

Deep learning is field of machine learning focused on complex architectures / networks for more complex problems. In the modern research and world, AI is now everywhere, every day more data is held on the internet and more complex models are developed and researched more now than ever. no 

2. Explain the architecture of a Convolutional Neural Network (CNN). Discuss the purpose of each layer in a typical CNN architecture and how they contribute to the network's ability to learn hierarchical representations from visual data.

The CNN architecture is better than the common FCN because of many things but the most important ones are first, it's way faster and second, how it was designed to be similar as how we humans see things and are processed in our brains. We have many important parts in here, first of all we have the **convolutional layers**, which are the responsibles for most of the job in there, those are the ones that will analyze the image spatial features in a matrix shape to extract features from it by the use of filters. Then pooling is used to reduce the size but remaining the most important information. Then we have the **rectified linear units** which work like electrical circuits as seen in class or as "brain sparks" which activate whenever they get some kind of excitement and give to the architecture non-linear complexity which enables our network to learn complex patterns between input and output. We then have a fully connected layer which is the neural network base if you ask me, which after getting those complex features and local representations of the image will connect them all to learn the final representation. Then there is the output layer which will be tied to some function like softmax to make the prediction. 

## Math Section:

Define the following terms commonly used in deep learning:
1. Activation function
2. Loss function
3. Gradient descent
4. Backpropagation

Given a dataset containing images of handwritten digits (e.g., MNIST dataset) with two classes (binary classification task - digits 0 and 1), design a simple Convolutional Neural Network (CNN) to classify the digits into their respective classes. The dataset is assumed to be linearly separable, meaning that there exists a decision boundary that can perfectly separate the two classes.
Derive the Mathematical Expression for the CNN Learning Rule:
Define the mathematical expression for updating the weights of the CNN to minimize the classification error. Explain how the CNN learning rule updates the weights based on the misclassification of each image.

Pichanga

## Cases to be Analyzed Section:

You are given a dataset of images containing handwritten digits. Design a CNN architecture to classify the digits into their respective classes (0-9). Discuss how you would preprocess the data, design the network architecture, and train the model.

preprocess:
Suppose its MNIST dataset, add few deg rotation, some upscaling and downscaling, noise for adversarial attacks and thats it.
Then go, conv relu pool conv relu pool flatten linear linear linear softmax.

Analyze a case study where a CNN model is used for image classification in the medical field. Discuss the dataset used, the network architecture, and the evaluation metrics used to assess the performance of the model.

Suppose its cancer detection through CV. The loss would have to minimize of False positives or false negatives. I would say false positives cuz cancer treatment if you don't have it or the shock after getting the notice isn't well handled and easy to solve after doing it by error. Then the network architecture depends a lot on the images and evaluation metrics just use a confusion matrix and then maybe F1 or ROC curve / AUC for more complex analysis.

## Code Review Section:

Review the following Python code snippet for training a CNN using PyTorch. Identify and correct any errors or potential issues in the code.

```py
import torch
import torch.nn as nn
import torch.optim as optim

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # in-chan 1, out-chan 16, kernel 3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # in-chan 16, out-chan 32, kernel 3
        self.fc1 = nn.Linear(32 * 7 * 7, 128) # 32 x 3 x 3 should be, then 128 is fine
        self.fc2 = nn.Linear(128, 10) # here is fine
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the dataset and dataloaders

# Define the loss function and optimizer

# Train the model
```

Implement a simple CNN architecture using TensorFlow/Keras for classifying images from the CIFAR-10 dataset. Provide the code for defining the model architecture, compiling the model, training the model on the dataset, and evaluating its performance on a test set.

