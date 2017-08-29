#!/usr/bin/env python3

from models import *
from utils import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

def vis_layer(activ_map):
    plt.subplot(121)
    plt.imshow(activ_map[:,:,0], cmap='gray')

def decon_img(layer_output):
    raw_img = layer_output.data.numpy()[0].transpose(1,2,0)
    img = (raw_img-raw_img.min())/(raw_img.max()-raw_img.min())*255
    img = img.astype(np.uint8)
    return img

def vis_decon(activ_map, decon_img):
    plt.subplot(121)
    plt.imshow(activ_map[:,:,0], cmap='gray')
    plt.subplot(122)
    plt.imshow(decon_img)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: '+sys.argv[0]+' img_file')
        sys.exit(0)

    img_filename = sys.argv[1]

    n_classes = 1000 # using ImageNet pretrained weights

    vgg16_c = VGG16_conv(n_classes)
    conv_layer_indices = vgg16_c.get_conv_layer_indices()

    img = np.asarray(Image.open(img_filename).resize((224,224)))
    img_var = torch.autograd.Variable(torch.FloatTensor(img.transpose(2,0,1)[np.newaxis,:,:,:].astype(float)))

    conv_out = vgg16_c(img_var)
    print('VGG16 model:')
    print(vgg16_c)

    plt.ion() # remove blocking
    plt.figure()
    vgg16_d = VGG16_deconv()
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
        activ_map = activ_map.transpose(1,2,3,0)
        activ_map_grid = vis_grid(activ_map)
        vis_layer(activ_map_grid)

        # only transpose convolve from Conv2d or ReLU layers
        conv_layer = layer
        if conv_layer not in conv_layer_indices:
            conv_layer -= 1
            if conv_layer not in conv_layer_indices:
                continue

        n_maps = activ_map.shape[1]

        done_act_maps = False
        while not done_act_maps:
            map_idx = input('Activation Map index (0-'+str(n_maps-1)+'): ')
            try:
                map_idx = int(map_idx)
            except ValueError:
                break

            decon = vgg16_d(vgg16_c.feature_outputs[layer][0][map_idx][None,None,:,:], conv_layer, map_idx, vgg16_c.pool_indices)
            img = decon_img(decon)
            vis_decon(activ_map_grid, img)
