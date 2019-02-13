import argparse
import torch

from PIL import Image
from torchvision.transforms import ToTensor
from model import Net

import numpy as np


# exporter settings
parser = argparse.ArgumentParser()
parser.add_argument('--model_in', type=str, default='super_resolution.pytorch')
parser.add_argument('--model_out', type=str, default='super_resolution.onnx')
parser.add_argument('--image', type=str, required=True, help='input image to use')

opt = parser.parse_args() 
print(opt)


# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))


# load the image
img = Image.open(opt.image)
img_to_tensor = ToTensor()
input = img_to_tensor(img).view(1, -1, img.size[1], img.size[0]).to(device)

print('input image size {:d}x{:d}'.format(img.size[0], img.size[1]))


# load the model
model = torch.load(opt.model_in).to(device)


# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('exporting model to ONNX...')
torch.onnx.export(model, input, opt.model_out, verbose=True, input_names=input_names, output_names=output_names)
print('model exported to {:s}'.format(opt.model_out))

