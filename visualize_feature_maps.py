import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data

from ssd.config import cfg
from ssd.data.datasets import MyDataset
from ssd.utils.checkpoint import CheckPointer
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.modeling.backbone import VGG

config = './configs/config.yaml'
image_input = cv2.imread('frame_75.jpg')
output_dir = './outputs/ssd_custom_coco_format'
result_file = './results/feature_maps_frame75.jpg'

class_name = {'__background__',
        'lubang', 'retak aligator', 'retak melintang', 'retak memanjang'}

cfg.merge_from_file(config)
cfg.freeze()
ckpt = None
device = torch.device('cpu')
model = build_detection_model(cfg)
model.to(device)

checkpoint = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
checkpoint.load(ckpt, use_latest=ckpt is None)
weight_file = ckpt if ckpt else checkpoint.get_checkpoint_file()
transforms = build_transforms(cfg, is_train=False)
model.eval()

conv_layers = []

model_children = list(model.children())
print(len(model_children))
print(type(model_children[0]))
print(type(model_children[1]))

counter = 0

for i in range(len(model_children)):
    if type(model_children[i]) == VGG:
        counter += 1
        conv_layers.append(model_children[i])

print(f'total convolutional layers: {counter}')
print(type(conv_layers))
print(len(conv_layers))

height, width = image_input.shape[:2]
image = transforms(image_input)[0].unsqueeze(0)

results = [conv_layers[0](image)]

for i in range(1, len(conv_layers)):
    results.append(conv_layers[i](results[-1]))

outputs = results

for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0]
    layer_viz = layer_viz.squeeze()
    print(layer_viz.size())

    for i, filter in enumerate(layer_viz):
        if i == 16: #because subplot(8, 8)
            break
        plt.subplot(4, 4, i+1)
        plt.imshow(filter.detach().numpy(), cmap='gray')
        plt.axis('off')

    print(f'Saving layer {num_layer} feature maps')
    plt.savefig(result_file)
    plt.close()
