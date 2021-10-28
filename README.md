# Fast Axiomatic Attribution for Neural Networks
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<img alt="PyTorch" height="20" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

This is the official repository accompanying the NeurIPS 2021 paper:

R. Hesse, S. Schaub-Meyer, and S. Roth. **Fast axiomatic attribution for neural networks**. _NeurIPS_, 2021, to appear.

[Paper](https://visinf.github.io/fast-axiomatic-attribution/) | [Preprint](https://visinf.github.io/fast-axiomatic-attribution/) | [Project Page](https://visinf.github.io/fast-axiomatic-attribution/) | [Video](https://visinf.github.io/fast-axiomatic-attribution/)

The repository contains:
- Pre-trained <img src="https://render.githubusercontent.com/render/math?math=\mathcal{X}">-DNN (X-DNN) variants of popular image classification models obtained by removing the bias term of each layer
- Detailed information on how to easily compute axiomatic attributions in closed form for your own project
- PyTorch code to reproduce the main experiments in the paper


## Pretrained Models

Removing the bias from different image classification models has a surpringly minor impact on the accuracy of the models while allowing to efficiently compute axiomatic attributions. Results of popular models with and without bias term (Regular vs. X-) on the ImageNet validation split are:

| Top-5 Accuracy| AlexNet | VGG16 | ResNet-50 |
|-------------|---------|-------|-----------|
| Regular DNN | 79.21   | 90.44 | 92.56     |
| X-DNN       | 78.54   | 90.25 | 91.12     |

You can **download** the pre-trained models from:

- AlexNet:
- VGG16
- ResNet-50
- X-AlexNet
- X-VGG16
- X-ResNet-50

## Using X-Gradient in Your Own Project

In the following we illustrate how to efficiently compute axiomatic attributions for X-DNNs. For a detailed example please see `demo.ipynb`. 

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
If the attribution is only used for evaluation you can set `create_graph` to `False`. If you want to use the attribution for training, e.g., for training with attribution priors, you can define `attribution_prior()` and update the weights of your model:
```python
loss1 = criterion(outputs, target) # standard loss
loss2 = attribution_prior(xgradient_attributions) # attribution prior    

loss = loss1 + lambda * loss2 # set weighting factor for loss2

optimizer.zero_grad()
loss.backward()
optimizer.step()
```
## Reproducing Experiments

The code and a README with detailed instructions on how to reproduce the results from experiments in Sec 4.1, Sec 4.2, and Sec 4.4. of our paper can be found in the [imagenet](imagenet) folder. To reproduce the results from the experiment in Sec 4.3. please refer to the [sparsity](sparsity) folder.

## Acknowledgment

We would like to thank the contributors of the following repositories for using parts of their publicly available code:
- https://github.com/suinleelab/attributionpriors
- https://github.com/ankurtaly/Integrated-Gradients
- https://github.com/pytorch/examples/tree/master/imagenet
- https://github.com/hongyi-zhang/Fixup



## Citation
If you find our work helpful please consider citing
```
@inproceedings{Hesse:2021:FAA,
  title     = {Fast Axiomatic Attribution for Neural Networks},
  author    = {Hesse, Robin and Schaub-Meyer, Simone and Roth, Stefan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  year      = {2021}
}
```
