<h1> Result of 1st DNN </h1>
[0.029716565811696637, 0.9907]

<h1> Definitions </h1>
<h2> Convolution </h2>
Convolution is process of getting new matrix (Z) from existing matrix (X) by passing another matrix (Y) over the existing matrix!

Example: X - 8x8x1, Y - 3x3x1. When we convolve Y on X, result - 6x6x1. Y is passed over X starting from top right corner of X. Each element of Y is multiplied with corresponding element of X; resulting element of Z is sum of all products. Y is then passed over X from immediate column (0 in image). As result of this process, Z is 6x6x1. Generalizing, if X - mxm and Y - nxn, then Z - (m-n+1)x(m-n+1).

![Image result for convolution image](https://i.stack.imgur.com/YDusp.png)

<h2> Filters/Kernels </h2>
Kernel is that matrix (Y) which passes over input matrix (X). Depth of kernel should be same as input. Height and width of kernel and input matrix can be different. Number of kernels will determine depth resulting matrix (Z).

![img](https://cdn-images-1.medium.com/max/1600/1*SvMakIpvPHJcElz-8S5swQ.png)

Stride is used to define number of pixel shift for kernel over input - stride = 2 means kernel is shifted 2 pixels at a time.

![img](https://cdn-images-1.medium.com/max/1600/1*nGHLq1hx0gt02OK4l8WmRg.png)

If kernel does not fit on input, padding is used on input to fit kernel.

![Image result for padding in cnn image](http://deeplearning.net/software/theano/_images/numerical_padding_strides.gif)

<h2> Epoch </h2>
Epoch means one run of entire dataset through the network. As entire dataset is always humongous, dataset is divided into batches. An iteration is number of batches required to complete an epoch. To enhance the learning of network entire dataset should be iteratively run through the network. This means multiple epochs need to be used. Very high number of epochs could overfit the data and very less epochs could underfit the data.

<h2> 1x1 Convolution </h2>
1X1 convolution is a kernel with depth same as input matrix. Convolving 1x1 on input keeps the height and width of the input same. With different number of 1x1 kernels, depth of output can be either increased or decreased or kept equal. Reducing the depth of output can reduce computational cost by reducing dimensions. Keeping the depth of output same will add non-linearity to input.

![img](https://qph.fs.quoracdn.net/main-qimg-0b3c4bbc86cc5c73efb8dbf2c699265a)

<h2> 3x3 Convolution </h2>
3x3 convolution is the most common type of convolution. It is used to extract edges/gradients, textures, patterns, parts of object, object and scene from an image. Convolving with 3x3 filter reduces the image by 2 units on both X and Y axes.

<h2> Feature Map </h2>
Feature map is the resulting matrix (Z). It gives information of those pixels which stand out loud during convolution. First feature map could give information on curves, consecutive one could give on circles.

Feature map will be different for every kernel used on the same input. Number of feature maps in the convolution layer depends on the number of kernels used.

Features => Textures => Part of object => Object => Patterns 

![img](https://ascelibrary.org/cms/attachment/4261df7b-cf05-4195-8bf7-16c61cdf40e8/)

<h2> Activation Function </h2>
Activation function is used to elevate or suppress the findings of features in an image. Relu (Rectified Linear Unit) is the most commonly used activation function.

<h2> Receptive Field </h2>
Receptive field are of two types - local RF and global RF. Local RF is the number of pixels a kernel can see in its immediate previous layer. Local RF is the size of the kernel used for convolution, this means that from a layer we can see 9 pixels of the immediate previous layer when using a 3x3 kernel. Global RF is the number of pixels of the input image which can be seen from a layer. Global RF increases by 2 for every layer of convolution done by 3x3 kernel. We should always aim to reach the size of the input image or more than that at the final layer as the global RF.
