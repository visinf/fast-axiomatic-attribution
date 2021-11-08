## Reproducing Results from Experiments in Sec 4.1, Sec 4.2, and Sec 4.4. (imagenet)

### Training
All the default hyperparameters in the code are set to the settings required to reproduce the results of our paper. We use a single NVIDIA A100 SXM4 40 GB GPU for training. To train your own models run the following commands. 

X-AlexNet & X-VGG16:
```
python train_alexnet_vgg.py    -a [model_type]\
                               --store_path [path_to_store]\
                               [path_to_imagenet]\
```
`[model_type]` can be `alexnet`, `xalexnet`, `vgg16`, or `xvgg16`.

X-ResNet-50:
```
python train_resnet.py   -a [model_type]\
                         --store_path [path_to_store]\
                         [path_to_imagenet] 
```
`[model_type]` can be `fixup_resnet50` or `xfixup_resnet50`.

### Evaluation
After training or downloading the pre-trained models, you can obtain numbers from the experiments in Sec 4.1. and 4.4. by running:
```
python val.py   --model_type [model_type]\
                --data_root [path_to_imagenet]\
                --model_dict [path_to_trained_model]\
                --accuracy\
                --compare_attribution\
                --evaluate_contrast\
```
Numbers from the experiments in Sec 4.2. can be obtained from:
```
python attribution_metrics.py   --model_type [model_type]\
                                --attr_method [attr_method]\
                                --data_root [path_to_imagenet]\ 
                                --model_dict [path_to_trained_models]\
```
`[model_type]` can be `alexnet`, `xalexnet`, `vgg16`, `xvgg16`, `fixup_resnet50`, or `xfixup_resnet50`.

`[attr_method]` can be `random`, `gradient`, `integrated_gradients128`, `input_x_gradient`, or `expected_gradients1`.

To visualize attributions run `visualize_attribution.ipnyb` (make sure to set the path for your model dictionary)
