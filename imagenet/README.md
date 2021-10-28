# Experiment in Section 4.1. - Removing the bias term in DNNs

All the default hyperparameters in our code are set to the settings required to reproduce our results. We use a single NVIDIA A100 SXM4 40 GB GPU for training. To reproduce the experiments please run the following commands:

To train the AlexNet and VGG models run:
```
python train_alexnet_vgg.py -a    [model_type]\
                                  --lr 0.01\
                                  [path_to_imagenet]\
```
[model_type] is in [alexnet, xalexnet, vgg16, xvgg16]

To train the ResNet models run:
```
python train_resnet.py -a   [model_type]\
                            [path_to_imagenet] 
```
[model_type] is in [fixup_resnet50, xfixup_resnet50]

After training, obtain the reported numbers by running:
```
python val.py   --model_type [model_type]\
                --data_root [path_to_imagenet]\
                --model_dict [path_to_trained_model]\
                --compare_attribution
```

[model_type] is in [alexnet, xalexnet, vgg16, xvgg16, fixup_resnet50, xfixup_resnet50]

# Experiment in Section 4.2. - Benchmarking gradient-based attribution methods

To reproduce results for a specific setup run:
```
python attribution_metrics.py   --model_type [model_type]\
                                --attr_method [attr_method]\
                                --data_root [path_to_imagenet]\ 
                                --model_dict [path_to_trained_models]\
```
[model_type] is in [alexnet, xalexnet, vgg16, xvgg16, fixup_resnet50, xfixup_resnet50]

[attr_method] is in [random, gradient, integrated_gradients128, input_x_gradient, expected_gradients1]

# Experiment in Section 4.4. - Homogeneity of X-DNNs

To reproduce results for decreasing contrasts run:
```
python val.py   --model_type [model_type]\
                --data_root [path_to_imagenet]\
                --model_dict [path_to_trained_models]\
                --evaluate_contrast\
```
[model_type] is in [alexnet, xalexnet]
