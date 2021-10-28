import json
import numpy as np
#import shap
#import shap.benchmark as benchmark
import scipy as sp
import math
from collections import OrderedDict
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
from tqdm import tqdm
from models import AlexNet, XAlexNet, vgg16, xvgg16, fixup_resnet50, xfixup_resnet50

import sys
from utils import AttributionPriorExplainer

import argparse

from captum.attr import IntegratedGradients, Saliency, InputXGradient

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=str, default='/data/datasets/imagenet/',
                        help="path to Dataset")
parser.add_argument("--model_dict", type=str, default='/data/imagenet_models/',
                        help="path to Dataset")
parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
parser.add_argument("--val_batch_size", type=int, default=2,
                        help='batch size for validation (default: 4)')
parser.add_argument("--model_type", type=str, default='alexnet',
                        choices=['alexnet', 'xalexnet', 'vgg16', 'xvgg16', 'resnet50', 'xresnet50'], help='model name')
parser.add_argument("--attr_method", type=str, default='gradient',
                        choices=['gradient', 'random', 'integrated_gradients128', 'input_x_gradient', 'expected_gradients1'], help='method name')


opts = parser.parse_args()
device = "cuda:" + str(opts.gpu_id)
torch.manual_seed(1)
np.random.seed(1)

if opts.model_type == 'alexnet':
    model = AlexNet().to(device)
    checkpoint = torch.load(opts.model_dict + 'alexnet_model_best.pth.tar', map_location=device)
elif opts.model_type == 'xalexnet':
    model = XAlexNet().to(device)
    checkpoint = torch.load(opts.model_dict + 'xalexnet_model_best.pth.tar', map_location=device)
elif opts.model_type == 'resnet50':
    model = fixup_resnet50().to(device)
    checkpoint = torch.load(opts.model_dict + 'resnet_model_best.pth.tar', map_location=device)
elif opts.model_type == 'xresnet50':
    model = xfixup_resnet50().to(device)
    checkpoint = torch.load(opts.model_dict + 'xresnet_model_best.pth.tar', map_location=device)
elif opts.model_type == 'vgg16':
    model = vgg16().to(device)
    checkpoint = torch.load(opts.model_dict + 'vgg16_model_best.pth.tar', map_location=device)
elif opts.model_type == 'xvgg16':    
    model = xvgg16().to(device)
    checkpoint = torch.load(opts.model_dict + 'xvgg16_model_best.pth.tar', map_location=device)

#remove preceding module. because it was stored as DataParallel
state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()

compare_attribution = True

for k, v in state_dict.items():
    if k[:7] == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

model.eval()

valdir = '/data/datasets/imagenet/val'
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_data = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
val_batch_size = opts.val_batch_size
val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=val_batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

unnormalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.weight = self.weight.to(device)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

def attribution_to_mask(attribution, percent_unmasked, sort_order, perturbation):

    attribution = attribution.clone().detach()
    attribution = attribution.mean(dim=1)
    attribution = attribution.unsqueeze(1)
    
    zeros = torch.zeros(attribution.shape).to(device)
    
    #invert attribution for negative case
    if sort_order == 'negative' or sort_order == 'negative_target' or sort_order == 'negative_topk':
        attribution = -attribution
        
    if sort_order == 'absolute':
        attribution = torch.abs(attribution)
    
    #for positive and negative the negative and positive values are always masked ond not considered for the topk
    positives = torch.maximum(attribution, zeros)
    nb_positives = torch.count_nonzero(positives)
    
    orig_shape = positives.size()
    positives = positives.view(positives.size(0), 1, -1)
    nb_pixels = positives.size(2)
    
    if perturbation == 'keep':
        # find features to keep
        ret = torch.topk(positives, k=int(torch.minimum(torch.tensor(percent_unmasked*nb_pixels).to(device), nb_positives)), dim=2)
        
    if perturbation == 'remove':
        #set zeros to large value
        positives_wo_zero = positives.clone()
        positives_wo_zero[positives_wo_zero == 0.] = float("Inf")
        # find features to keep
        ret = torch.topk(positives_wo_zero, k=int(torch.minimum(torch.tensor(percent_unmasked*nb_pixels).to(device), nb_positives)), dim=2, largest=False)
    ret.indices.shape   
    # Scatter to zero'd tensor
    res = torch.zeros_like(positives)
    res.scatter_(2, ret.indices, ret.values)
    res = res.view(*orig_shape)

    res = (res == 0).float() # set topk values to zero and all zeros to one
    res = res.repeat(1,3,1,1)
    return res

torch.manual_seed(1)
np.random.seed(seed=1)

mode = opts.attr_method

print('Starting: ', mode)
sort_orders = ['positive', 'negative', 'absolute']
perturbations = ['keep', 'remove']

options = list(itertools.product(sort_orders, perturbations))

percent_unmasked_range = np.geomspace(0.01, 1.0, num=16)

smoothing = GaussianSmoothing(3, 51, 41)

vals = {}
plot_imgs = []

for sort_order, perturbation in options:   
    for percent_unmasked in percent_unmasked_range:
        vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)] = 0.0

    
if mode == 'integrated_gradients128':
    ig = IntegratedGradients(model)
    baseline = torch.zeros((1,3,224,224)).to(device)

if mode == 'expected_gradients1':
    APExp = AttributionPriorExplainer(val_data, batch_size=1,k=1, scale_by_inputs=True)
 
if mode == 'gradient':
    saliency = Saliency(model)
    
if mode == 'input_x_gradient':
    input_x_gradient = InputXGradient(model)
    
    
itrs = 0
for i, (images, targets) in tqdm(enumerate(val_loader)):
    images = images.to(device)
    
    #images.requires_grad = True
    targets = targets.to(device)
    
    #out = model(images)
    
    # get attribution for sample
    if mode == 'gradient':
        attribution = saliency.attribute(images, target=targets, abs=False)
        attribution = attribution.to(device).float().detach()
    elif mode == 'random':
        attribution = torch.rand(images.shape).to(device) - 0.5 # center around 0
    elif mode == 'integrated_gradients128':
        attribution = ig.attribute(images,
                                target = targets,
                                baselines=baseline,
                                method='gausslegendre',
                                n_steps=128,
                                return_convergence_delta=False)
        attribution = attribution.to(device).float().detach()
    elif mode == 'expected_gradients1':
        attribution = APExp.shap_values(model,images, sparse_labels=targets)
        attribution = attribution.to(device).float().detach()
    elif mode == 'input_x_gradient':
        attribution = input_x_gradient.attribute(images, target=targets)
        attribution = attribution.to(device).float().detach()
    
    masks = []
    for sort_order, perturbation in options:
        for percent_unmasked in percent_unmasked_range:
            #create masked images
            for sample in range(attribution.shape[0]):
                mask = attribution_to_mask(attribution[sample].unsqueeze(0), percent_unmasked, sort_order, perturbation)
                masks.append(mask)

    mask = torch.cat(masks, dim=0)

    images_masked_pt = images.clone().repeat(int(mask.shape[0]/val_batch_size), 1, 1, 1)
    images_smoothed_pt = images.clone().repeat(int(mask.shape[0]/val_batch_size), 1, 1, 1)
    images_smoothed_pt = F.pad(images_smoothed_pt, (25,25,25,25), mode='reflect')
    images_smoothed_pt = smoothing(images_smoothed_pt)
    images_masked_pt[mask.bool()] = images_smoothed_pt[mask.bool()]

    #images_masked = normalize(torch.tensor(images_masked_np / 255.).unsqueeze(0).permute(0,3,1,2))
    images_masked = images_masked_pt
    images_masked = images_masked.to(device)
    out_masked = model(images_masked)

    #split out_masked in the chunks that correspond to the individual run
    option_runs = torch.split(out_masked, int(out_masked.shape[0]/len(options)))
    for o, (sort_order, perturbation) in enumerate(options):
        option_run = option_runs[o] # N, 1000
        percent_unmasked_runs = torch.split(option_run, int(option_run.shape[0]/len(percent_unmasked_range)))  # N, 1000
        for p, percent_unmasked in enumerate(percent_unmasked_range):
            percent_unmasked_run = percent_unmasked_runs[p] # N, 1000
            #if len(percent_unmasked_run.shape) == 1:
            #    percent_unmasked_run = percent_unmasked_run.unsqueeze(0)

            if sort_order == 'positive':
                vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)] += torch.gather(percent_unmasked_run, 1, targets.unsqueeze(-1)).sum().item()
            if sort_order == 'negative':
                vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)] += torch.gather(percent_unmasked_run, 1, targets.unsqueeze(-1)).sum().item()
            if sort_order == 'absolute':
                correct = (torch.max(percent_unmasked_run, 1)[1] == targets).float().sum()
                vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)] += correct

    itrs += 1


for sort_order, perturbation in options:   
    for percent_unmasked in percent_unmasked_range:
        vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)] /= (itrs*val_batch_size)

for sort_order, perturbation in options:
    xs = []
    ys = []
    for percent_unmasked in percent_unmasked_range:
        xs.append(percent_unmasked)
        ys.append(vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)])
    auc = np.trapz(ys, xs)
    print(mode)
    print('AUC for ' + perturbation + "_" + sort_order + ": ", auc)

print('FINNISHED')
