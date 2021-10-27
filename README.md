# Fast Axiomatic Attribution for Neural Networks

This repository contains:
- code to reproduce the main experiments in "Fast Axiomatic Attribution for Neural Networks" 
- Pre-trained X-DNN variants of popular image classification models obtained by removing the bias term of each layer
- Detailed information on how to easily compute axiomatic attributions in closed form in your own project

### Paper

### Pretrained Models

Removing the bias from different image classification models has a surpringly minor impact on the accuracy of the models while allowing to efficiently compute axiomatic attributions. Results of popular models with and without bias term on the ImageNet validation split are:

| Top-5 Accuracy| AlexNet | VGG16 | ResNet-50 |
|-------------|---------|-------|-----------|
| Regular DNN | 79.21   | 90.44 | 92.56     |
| X-DNN       | 78.54   | 90.25 | 91.12     |

#### Download

- AlexNet:
- VGG16
- ResNet-50
- X-AlexNet
- X-VGG16
- X-ResNet-50

### Using X-Gradient in Your Own Project

### Reproducing Experiments


<img src="https://render.githubusercontent.com/render/math?math=\mathcal{X}">-DNNs
