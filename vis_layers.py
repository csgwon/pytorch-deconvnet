#!/usr/bin/env python3

from models import *
from utils import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

def vis_layer(activ_map):
    plt.ion()
    plt.imshow(activ_map[:,:,0], cmap='gray')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: '+sys.argv[0]+' img_file')
        sys.exit(0)

    img_filename = sys.argv[1]

    n_classes = 1000 # using ImageNet pretrained weights

    vgg16_c = VGG16_conv(n_classes)

    img = np.asarray(Image.open(img_filename).resize((224,224)))
    img_var = torch.autograd.Variable(torch.FloatTensor(img.transpose(2,0,1)[np.newaxis,:,:,:].astype(float)))

    conv_out = vgg16_c(img_var)
    print('VGG16 model:')
    print(vgg16_c)

    done = False
    while not done:
        layer = input('Layer to view: ')
        try:
            layer = int(layer)
        except ValueError:
            continue
            
        if layer < 0:
            sys.exit(0)
        activ_map = vgg16_c.feature_outputs[layer].data.numpy()
        vis_layer(vis_grid(activ_map.transpose(1,2,3,0)))
