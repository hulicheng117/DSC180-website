---
title: New Initialization Mechanisms for Convolutional Neural Networks
feature_image: "https://www.world-wide.org/images/dl.jpg"
author: Keyu Long, Tiffany Yu, Grace Lam, Daniel Shi, Licheng Hu

---

### Introduction
Neural networks, mirroring the human brain’s pattern recognition abilities, have revolutionized machine learning. Their success spans various fields, from healthcare to finance,
showcasing exceptional problem-solving and data interpretation capabilities. As a neural net-
works increasingly outperform traditional algorithms, it becomes crucial to unravel their
complex learning mechanisms.
Previous studies Beaglehole et al. (2023) and Radhakrishnan et al. (2023) formulated Convolutional Neural Feature Ansatz, demonstrating that features selected by convolutional
networks can be recovered by computing the average gradient outer product (AGOP) of
the trained network with respect to image patches given by empirical covariance matrices
of filters at any given layer. Concurrently, these investigations identified an Average Gradient
Outer Product (AGOP) and Neural Feature Matrix (NFM) as key elements characterizing
feature learning in neural networks.
Meanwhile, another critical aspect of deep learning that influences neural network performance-
mance and convergence is the method of initialization. Proper Initialization is crucial; due
to the use of backpropagation in neural networks, improper initialization can lead to the
vanishing or exploding gradient problem, thereby affecting the overall training process.
Our study aims to combine the concepts of NFM and AGOP with initialization methods.
This exploration seeks to address the question: How does the application of the Neural
Feature Matrix and Average Gradient Outer Product as initialization affect the performance of neural networks?

### Feature Learning with NFM(Neural Feature Matrix) and AGOP(Average Gradient Outer Product)
* __The Neural Feature Matrix (NFM)__ is the neural feature matrix resulting from multiplying model’s weight matrices.
* __The Average Gradient Outer Product (AGOP)__ is the average gradient outer product over patches, it is the gradient
with respect to that patch average over data.
* Previous studies posit the __Convolutional Neural Feature Ansatz__ [1], which states that there is a positive correlation between AGOP and NFM.
<div style="text-align:center">
    <img width="392" alt="Screenshot 2024-03-14 at 4 24 55 PM" src="https://github.com/hulicheng117/DSC180-website/assets/97436268/b66bdc60-9053-4b5c-921d-8361693e0ca6">

</div>

* The significance of AGOP and NFM is highlighted by findings suggesting that
these measures, of early layers, perform operations similar to edge detection.
As shown in Figure 1, Patch-AGOPs and NFMs from pre-trained VGG11
identified edges in images and progressively highlighted regions of images
used for prediction

![output](https://github.com/hulicheng117/DSC180-website/assets/97436268/eb8d1da0-e817-4e4c-bbdb-fdccec8fe652)


### Dataset
To investigate the application of the Neural Feature Matrix (NFM) and Average Gradient Outer Product (AGOP) as initialization methods, we will be examining their performance across four different datasets: SVHN, CIFAR-10, CIFAR-100, and Tiny ImageNet.
* __SVHN__ (Street View House Numbers) is a dataset that contains a total of 600,000 digit images from Google Street View. Each image is of the size 32x32 pixels in size.
* __CIFAR-10__ (Canadian Institute for Advanced Research, 10 classes) consists of 60000 32x32 color images in 10 classes, with 6000 images per class.     
* __CIFAR-100__ (Canadian Institute for Advanced Research, 100 classes) is an extension of CIFAR-10. Instead of 10 classes, it has 100 classes.
* __Tiny-Imagenet__ is a subset of the ImageNet dataset. It contains 100000 images of 200 classes (500 for each class) downsized to 64×64 colored images 

### Methods Overview
 * __Model__: We used a VGG11 model which has 8 convolutional layers followed by 3 fully connected layers.
    - __Method__: Investigate the performance of AGOP and NFM initialization compared to other common initializations by examining accuracy and loss.
    - __Initialization Methods__:
        - __Normal__:  extract weights from a normal distribution.
        - __Uniform__: extract weights from a uniform distribution.
        - __AGOP and NFM__:  We use the pre-trained model from PyTorch to extract AGOP and NFM, and use these as the covariant matrix to generate a multivariate Gaussian for sampling weights for initialization.
        - __Kaiming__: Kaiming initialization[2] is one of the most effective methods for initializing convolutional layer weights, it is also the default initialization method for Conv2d in PyTorch. The method changes the parameters (e.g. std for normal, range for uniform, covariance matrix NFM) of the distribution based on layer width and the type of activation function.
  * __Experiment and Hyper-parameters__: Using the VGG11 architecture, we investigated the performance of AGOP and NFM initialization methods on model training. We employed SGD with a learning rate of 0.001 and cross-entropy loss. The goal was to control initialization methods as the only variable.



### Training graphs
![cifar_100_acc](https://github.com/hulicheng117/DSC180-website/assets/97436268/8e934d9d-6f8f-4f29-be9f-0b8f1a5777f1)
<div style="text-align:center">
   <img style="text-align:center" src="https://github.com/hulicheng117/DSC180-website/assets/97436268/7def6247-30fa-4ee8-ac41-882ee93613fa">
</div>



In our training graph we can see both training and validation accuarcy of Kaiming_NFM outperform the Kaiming_uniform which is the default initialization
method used in pytorch on CIFAR-100. We also ploted the difference of validation loss and training loss for each initialization method acrossed training process. When the difference is negative, the validation loss is lower than the training loss. This can be a sign of underfitting, meaning the model is not capturing the underlying trends in the training data enough. When the difference is positive (above 0 on the y-axis), the validation loss is higher than the training loss and indicate overfitting. From the graph, our initialization method of agop and nfm showed robust to overfit than other initialization
method and Kaiming_nfm also outperformed among all initialization methods with Kaiming initialization scaling.



### Results and Disscussion
<div style="text-align:center">
      <img width="1060" alt="table" src="https://github.com/hulicheng117/DSC180-website/assets/97436268/bfdd938d-474d-4c0f-869a-bf3f855441a3">
</div>


NFM initialization outperformed all other initialization methods, achieving the highest validation accuracy across all the datasets. With Kaiming NFM, in particular, achieving remarkable results in 3 of the 4 datasets. It was able to attain highest validation accuracies of 93.23% on SVHN, 48.33% on CIFAR100, and 33.78% on Tiny-ImageNet. 

Our initialization methods can be viewed as a soft way of transferring learning. We used the pre-trained model on ImageNet and train the model on simpler datasets. It is worth investigating what if we initialize with model pre-trained on simpler datasets and train on more complex datasets. Some other potential areas of improvement include:

  - Perform hyperparameter tuning to obtain the best performance of each method.
  - Try to repeat training the model and extracting AGOP/NFM to see if it can improve the performance.
  - Try to expand the method to transformers.

### Conclusion
In conclusion, our study observed the impact of advanced initialization methods on the
performance of neural networks, with a specific focus on the Neural Feature Matrix (NFM)
and Average Gradient Outer Product (AGOP). The investigation centered on the application
of these methods to the VGG11 model, trained on a range of datasets from SVHN to Tiny
ImageNet.

Our findings suggest that the integration of NFM and AGOP with traditional initialization
methods can lead to substantial improvements in validation accuracy. Notably, the Kaiming
NFM initialization outperformed standard practices in several datasets, marking a signifi-
cant step forward in neural network training strategies. The implications of this advance-
ment are profound, offering potential enhancements in various applications where deep
learning models are pivotal.


### Acknowledgement and References
- We would like thank our mentors, Misha Belkin, Yian Ma, for their invaluable
guidance and support throughout this project.
- Special thanks to Daniel Beaglehole for his guidance.His insights have been
instrumental in shaping the success of this project.
- Special thanks to Licheng’s pet, Xiaohei, for providing her picture for our
visualization.

[1] Daniel Beaglehole, Adityanarayanan Radhakrishnan, Parthe Pandit, and Mikhail Belkin. Mechanism of feature learning in convolutional neural networks, 2023.

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification, 2015.

[3] Adityanarayanan Radhakrishnan, Daniel Beaglehole, Parthe Pandit, and Mikhail Belkin.Mechanism for feature learning in neural networks and backpropagation free machine learning models. Science, 0(0).



