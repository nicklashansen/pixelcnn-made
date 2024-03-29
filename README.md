# PyTorch implementation of PixelCNN-MADE

This is a PyTorch implementation of the auto-regressive PixelCNN-MADE architecture, which mixes ideas from the two papers [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) (2016) and [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) (2015) to produce colored 28x28 MNIST digits. Pixel intensities have been quantized to 2 bits (i.e. four intensities for each color channel). The dataset can be downloaded from [here](https://drive.google.com/open?id=1hm077GxmIBP-foHxiPtTxSNy371yowk2).

## Training data

I have used 60,000 quantized 28x28 images from the colored MNIST dataset. Here's a few samples from the training set:

![TrainingSet](https://i.imgur.com/A4gI8LS.png)


## Examples

After 50 epochs of training, a network consisting of 12 residual blocks - see [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (2015) - followed by a 3-layer MADE generates samples like the following:

![Examples](https://i.imgur.com/wkLYtx9.png)

You are very welcome to extend the code however you like. If you produce anything cool, be sure to let me know!
