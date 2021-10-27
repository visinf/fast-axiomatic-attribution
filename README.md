# Fast Axiomatic Attribution for Neural Networks
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<img alt="PyTorch" height="20" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

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

In the following we illustrate how to efficiently compute axiomatic attributions for X-DNNs. For a detailed example please see `demo.py`. 

First, make sure that `requires_grad` of your input is set to `True` and run a forward pass:
```python
inputs.requires_grad = True

# forward pass
outputs = model(inputs)
```
Next, you can compute X-Gradient via:
```python
# compute attribution
target_outputs = torch.gather(outputs, 1, target.unsqueeze(-1))
gradients = torch.autograd.grad(torch.unbind(target_outputs), inputs, create_graph=True)[0] # set to false if attribution is only used for evaluation
xgradient_attributions = gradients * images
```
If the attribution is only used for evaluation you can set `create_graph` to False. Finally, you can define `attribution_prior()` and update the weights of your model:
```python
loss1 = criterion(outputs, target) # standard loss
loss2 = attribution_prior(xgradient_attributions) # attribution prior    

loss = loss1 + lambda * loss2 # set weighting factor for loss2

optimizer.zero_grad()
loss.backward()
optimizer.step()
```


### Reproducing Experiments






If the attributions are only used for evaluation they can easily be computed with:

```python
inputs.requires_grad = True # make sure requires_grad of your input is set to True

# forward pass
outputs = model(inputs)

# compute attribution
target_outputs = torch.gather(outputs, 1, target.unsqueeze(-1))
gradients = torch.autograd.grad(torch.unbind(target_outputs), inputs, create_graph=False)[0]
xgradient_attributions = gradients * images
```
When using X-Gradient for training with attribution priors you need to define `attribution_prior()` and you need to make sure that create_graph=True for the grad() function:
```python
inputs.requires_grad = True # make sure requires_grad of your input is set to True

# forward pass
outputs = model(inputs)

# compute attribution
target_outputs = torch.gather(outputs, 1, target.unsqueeze(-1))
gradients = torch.autograd.grad(torch.unbind(target_outputs), inputs, create_graph=True)[0] # set create_graph to True
attributions = gradients * images
loss1 = criterion(outputs, target)
loss2 = attribution_prior(attributions)    

loss = loss1 + lambda * loss2 # set weighting factor for loss2

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{X}">-DNNs
