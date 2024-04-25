# Transfer Learning

In practice few people really train networks with from scratch (random weight initialization). This, because it's relatively rare to find a big enough dataset of sufficient size. It's common to train a Net with ImageNet which has more than a million images and then use the Net as initialization or fixed feature extractor for the task of interest.

There are two major transfer learning escenarios:

1. **Fine-tuning the ConvNet:** We initialize the net with a pre-trained network then the rest of the training is as usual. 
> **Now understood, my explanation:** This is way easier than the other escenario, just do forward and back propagation again but with the new data. What is a bit more complex is what weights to alter, you could change them all, but the first ones usually contain generic features such as edge detectors, blob color detectors, etc. So it may be a better choice to change weights of the half-last layers or something like that.

2. **ConvNet as feature extractor:** Here we freeze the weights except for the fully connected layer at the end. That one is then replaced with a new one with random weights and only that layer is trained. 
> **Now understood, my explanation:** Instead of doing the forward propagation and making the predictions in the last layer, we will do the forward propagation and stop before doing the predictions. This will give us (before preds) a n-Dim vector which will contain that extracted image features and we will then use those to train another classifier (SVM for example) to test the new dataset. Btw, its important that the extracted image features are ReLU'd.

Now, how do you decide what type of transfer-learning you should perform on a new dataset? This depends on many factors but the two most important ones are, how big is the new dataset and how similar it is to the original dataset. Lets explore some escenarios:

1. **Small and similar to the original dataset.** Since its few data its not a good idea to fine-tuning on it cause it can led to overfitting problems. The best idea would be to train a linear classifier on it.
2. **Large and different to the original dataset.** Since its a good amount of data, we can fine-tuning and won't overfit.
3. **Small and different to the original dataset.** Its few data so we can't fine-tune the other net. Training a linear classifier on top of the other net also isn't a good idea since the data isn't similar so it will probably have more dataset-specific features, however what we can do is grab some of the first layers which have more generic features and use that as a feature extractor.
4. **Large and similar to the original dataset.** We could say to train the model from zero since the dataset is big enough for it however, in practice this isn't done neither. We can initialize the model weights from a pre-trained model and then fine-tuning without worrying about overfitting.

### Advices or some things to keep in mind.

- Constraints from pretrained models: If you wish to use a pre-trained model you are slightly constrained by the model architecture for the new dataset. However, some changes are valid, for example: you can easily run a pretrained network on different spatial size images. This happens because Conv/Pool layers forward function's is independent of the input volume spatial size. In case of FC layers. (For example in AlexNet the final pooling volume before the first FC layer is of size [6x6x512]. Therefore, the FC layer looking at this volume is equivalent to having a Convolutional Layer with 6x6 size receptive field and with 0 padding.)

- Learning rates: It's common to use a smaller learning rate on fine tuning than when doing the random initialized weights for the new linear classifier that will compute the class scores of the new dataset. This is because we assume the weights from the ConvNet are already pretty good so we don't want to distort them too quickly and too much.

[Razavian, A. Azizpour, H. Sullivan, J & Carlson, S. (March, 2014.) CNN Features off-the-shelf: an Astounding Baseline for Recognition](https://arxiv.org/pdf/1403.6382.pdf) trains SVMs on features from ImageNet-pretrained ConvNet and reports several state of the art results.

[Donahue, J. Yangqinq, J. Vinyals, O. Hoffman, J. Zhang, N. Tzeng, E & Darrell, T. (October, 2013.) DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition] reported similar findings in 2013. The framework in this paper (DeCAF) was a Python-based precursor to the C++ Caffe library.

[Yosinsky, J. Clune, J. Bengio, Y. & Lipson, H. (November, 2014). How transferable are features in deep neural networks?](https://arxiv.org/pdf/1411.1792.pdf) studies the transfer learning performance in detail, including some unintuitive findings about layer co-adaptations.