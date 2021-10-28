# script to get numbers for Experiment 1 and 4

# Experiment 1:
# python val_accuracy_diff.py --gpu_id 2 --compare_attribution --model_type resnet50 --val_batch_size 2
# Experiment 4:
# python val_accuracy_diff.py --gpu_id 2 --evaluate_contrast --model_type alexnet --val_batch_size 256
# python val_accuracy_diff.py --gpu_id 3 --evaluate_adversarial --model_type alexnet --val_batch_size 128

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse
import os
from collections import OrderedDict

from captum.attr import InputXGradient, IntegratedGradients

from models import AlexNet, XAlexNet, vgg16, xvgg16, fixup_resnet50, xfixup_resnet50

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=str, default='/path/to/imagenet/',
                        help="path to ImageNet")
parser.add_argument("--model_dict", type=str, default='/data/imagenet_models/',
                        help="path to trained models")
parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
parser.add_argument("--val_batch_size", type=int, default=16,
                        help='batch size for validation (default: 16)')
parser.add_argument("--model_type", type=str, default='alexnet',
                        choices=['alexnet', 'xalexnet', 'vgg16', 'xvgg16', 'fixup_resnet50', 'xfixup_resnet50'], help='model name')
parser.add_argument("--compare_attribution", action='store_true', default=False,
                        help="compute the attribution difference of IG and InputXGradient resp. X-Grad")
parser.add_argument("--evaluate_contrast", action='store_true', default=False,
                        help="compute the top-1 accuracy for different contrasts")
parser.add_argument("--accuracy", action='store_true', default=False,
                        help="compute the top-1 and top-5 accuracy on the ImageNet validation split")

opts = parser.parse_args()
device = "cuda:" + str(opts.gpu_id)
torch.manual_seed(1)
np.random.seed(1)

valdir = os.path.join(opts.data_root, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.val_batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

def get_nr_correct_classifications_topk(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        result = 0
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            result += correct_k
            res.append(correct_k.mul_(100.0 / batch_size))
        return result

if opts.model_type == 'alexnet':
    model = AlexNet().to(device)
    checkpoint = torch.load(opts.model_dict + 'alexnet_model_best.pth.tar', map_location=device)
elif opts.model_type == 'xalexnet':
    model = XAlexNet().to(device)
    checkpoint = torch.load(opts.model_dict + 'xalexnet_model_best.pth.tar', map_location=device)
elif opts.model_type == 'fixup_resnet50':
    model = fixup_resnet50().to(device)
    checkpoint = torch.load(opts.model_dict + 'fixup_resnet50_model_best.pth.tar', map_location=device)
elif opts.model_type == 'xfixup_resnet50':
    model = xfixup_resnet50().to(device)
    checkpoint = torch.load(opts.model_dict + 'xfixup_resnet50_model_best.pth.tar', map_location=device)
elif opts.model_type == 'vgg16':
    model = vgg16().to(device)
    checkpoint = torch.load(opts.model_dict + 'vgg16_model_best.pth.tar', map_location=device)
elif opts.model_type == 'xvgg16':    
    model = xvgg16().to(device)
    checkpoint = torch.load(opts.model_dict + 'xvgg16_model_best.pth.tar', map_location=device)

#remove preceding module. because it was stored as DataParallel
state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()


for k, v in state_dict.items():
    if k[:7] == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)


model.eval()

if opts.accuracy:
    correct_top1 = 0
    correct_top5 = 0
    print('Total itrs: ', len(val_loader))
    for i, (images, targets) in tqdm(enumerate(val_loader)):
        images = images.to(device)
        baselines = torch.zeros(images.shape).to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        
        correct_top1 += get_nr_correct_classifications_topk(outputs, targets, topk=(1,)).item()
        correct_top5 += get_nr_correct_classifications_topk(outputs, targets, topk=(5,)).item()

    print('Accuracy top-1: ', correct_top1 / (len(val_loader) * opts.val_batch_size))
    print('Accuracy top-5: ', correct_top5 / (len(val_loader) * opts.val_batch_size))


if opts.compare_attribution:
    ig = IntegratedGradients(model, multiply_by_inputs=True)
    input_x_gradient = InputXGradient(model)

    total_distance = 0.
    print('Total itrs: ', len(val_loader))
    for i, (images, targets) in tqdm(enumerate(val_loader)):
        images = images.to(device)
        baselines = torch.zeros(images.shape).to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        
        attributions1, approximation_error = ig.attribute(images,
                                                            baselines,
                                                            target=targets,
                                                            method='gausslegendre',
                                                            return_convergence_delta=True,
                                                            n_steps = 128)

        attributions2 = input_x_gradient.attribute(images, target=targets)

        distance = torch.mean(torch.abs(attributions1 - attributions2)) / (torch.mean(torch.abs(attributions1)) + 10e-8)
        total_distance = total_distance + distance.item()

    print('Distance: ', total_distance / len(val_loader))
   
if opts.evaluate_contrast:
    
    alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for alpha in alphas:
        correct_top1 = 0
        for i, (images, targets) in tqdm(enumerate(val_loader)):
            images = images.to(device) 
            images = images * alpha
            targets = targets.to(device)
            
            outputs = model(images)
            
            correct_top1 += get_nr_correct_classifications_topk(outputs, targets, topk=(1,)).item()
        print('Accuracy top-1 for alpha ', alpha, correct_top1 / (len(val_loader) * opts.val_batch_size))
